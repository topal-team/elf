'''
Core pipeline objects
'''

import os
import torch
import torch.distributed as dist
from .schedule import *
from .engine import Engine
from .utils import Timer, TensorMetadata, activations_offloading
from .partitioners import partition_graph, get_inputs_outputs_single, create_subgraph
from collections import deque
import logging
logger = logging.getLogger("pipeline")

class PipelineBlock():
    '''
    Pipelines are made up of sequential blocks, numbered [0..n]
    Each block is one layer or group of contiguous layers placed on one device
    '''
    def __init__(self, model, id_, placement, inputs, outputs, dp = 1):
        '''
        :param model: layer / group of layers that will perform the computation
        :type model: nn.Module
        :param id_: number of this block in the pipeline
        :type id_: int
        :param placement: mapping of each id on a device
        :type placement: List[int]
        :param inputs: name of variables taken as input by the block
        :type inputs: List[str]
        :param outputs: name of variables returned by the block
        :type outputs: List[str]
        '''
        super(PipelineBlock, self).__init__()
        # Block infos
        self.rank = placement[id_] # global rank
        self.model = model.cuda() if torch.cuda.is_available() else model
        
        if dp > 1:
            ws = int(os.getenv('WORLD_SIZE'))
            dppg = dist.new_group([i for i in range(ws) if (i % len(placement)) == (rank % len(placement))])
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, process_group = dppg)
            
        self.id = id_ # rank in the model
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

        # Queues of tensors to process
        # Structure of one element is {variable1: [Work, Tensor], variable2: [Work, Tensor], ..}
        self.inputs_to_forward = deque() # Waiting for forward
        self.grads_to_backward = deque() # Waiting for backward

        # Structure of one element is {variable1: Tensor, variable2: Tensor, ..}
        self.act_to_send = deque() # Sent to next block
        self.grads_to_send = deque() # Sent to previous block
        self.act_to_keep = deque() # Kept for backward
        self.inputs_to_keep = deque() # Kept for backward

        # Ranks where the previous/next blocks in the model are placed
        self.previous = None if self.id == 0 else placement[self.id - 1]
        self.next = None if self.id == len(placement) - 1 else placement[self.id + 1]

        self.metadata = {}
        self.out_metadata = {}

        self.compute_time = 0 # used to measure idle time

        self.inputs = sorted(inputs) # name of input variables
        self.outputs = sorted(outputs) # name of output variables
        # sorted alphabetically to make sure the order is consistent across devices
        # note: can this order matter ? can it be faster to communicate in some order depending on the tensor shapes/sizes ?

    def __str__(self) -> str:
        return f'[Layer {self.id} : GPU {self.rank}]'

    def forward(self, **options):
        '''
        Perform the forward pass for one tensor of activations and register it as computed

        :param **options: options to modify the forward behaviour
        :return: if this is the last block of the pipeline, returns its output. Otherwise returns None
        :rtype: Tensor or None
        '''
        logger.debug(f'{self} - Computing one forward with options {options}')
        
        # Wait for all communications to finish
        x = self.inputs_to_forward.popleft()
        for key in self.inputs:
            work, i = x[key]
            if work is not None: work.wait()
            x[key] = i
        
            if x[key].dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                x[key].requires_grad = True

        with Timer() as timer:
            if  options.get('remat'):
                with torch.no_grad(): y = self.model(**x)
            elif options.get('offload'):
                with activations_offloading():
                    y = self.model(**x)
                self.act_to_keep.append(y)
                # x = x.cpu()
            else:
                y = self.model(**x)
                self.act_to_keep.append(y)
                
        self.compute_time += timer.time()
           
        self.act_to_send.append(y)
        self.inputs_to_keep.append(x)

        if self.next is None:
            return self.act_to_send.popleft()
        
    def backward(self, **options):
        '''
        Perform the backward pass for one tensor of gradients and register it as computed
        Backward assumes activations AND grads to be on top of the queue

        :param **options: options to modify the backward behaviour
        '''
        logger.debug(f'{self} - Computing one backward with options {options}')

        x = self.inputs_to_keep.popleft()

        if options.get('remat'):
            with Timer() as timer:
                act = self.model(**x)
            self.compute_time += timer.time()
        elif options.get('offload'):
            act = self.act_to_keep.popleft().cuda()
            # x = x.cuda()
        else:
            act = self.act_to_keep.popleft()
            
        # Wait for all communications to finish
        grads = self.grads_to_backward.popleft()
        for key in self.outputs:
            work, g = grads[key]
            if work is not None: work.wait()
            grads[key] = g

        with Timer() as timer:
            for key in self.outputs:
                # Perform a backward pass for each output tensor; once the last one is done, the graph can be freed
                act[key].backward(grads[key], retain_graph = (key != self.outputs[-1]))
        self.compute_time += timer.time()

        self.grads_to_send.append({key: value.grad.data for key, value in x.items() if value.requires_grad})
            
    def send_forward(self, **options):
        '''
        Send one activation to the next layer in the model

        :param **options: options to modify the send behaviour
        :return: If the communications needs to be batched, returns them
        :rtype: List[dist.P2POp] or None
        '''
        dst = options.get("dst") or self.next
        if dst is None or dst == self.rank: return

        activations = self.act_to_send.popleft()

        if options.get("batch"):
            return [dist.P2POp(dist.isend, a, dst) for a in activations.values()]
        else:
            logger.debug(f'{self} - Sending activations to layer {self.id + 1} on rank {dst}')
            for a in activations.values():
                dist.isend(a, dst)

    def send_backward(self, **options):
        '''
        Send one gradient tensor to the previous layer in the model

        :param **options: options to modify the send behaviour
        :return: If the communications needs to be batched, returns them
        :rtype: List[dist.P2POp] or None
        '''
        dst = options.get("dst") or self.previous
        if dst is None or dst == self.rank: return

        grads = self.grads_to_send.popleft()

        if options.get("batch"): 
            return [dist.P2POp(dist.isend, g, dst) for g in grads.values()]
        else:
            logger.debug(f'{self} - Sending gradients to layer {self.id - 1} on rank {dst}')
            for g in grads.values():
                dist.isend(g, dst)

    def recv_forward(self, mb_size, **options):
        '''
        Receive and store one activation to forward

        :param mb_size: size of the micro batch to receive
        :type mb_size: int
        :param **options: options to modify the send behaviour
        :return: If the communications needs to be batched, returns them
        :rtype: List[dist.P2POp] or None
        '''
        src = options.get("src") or self.previous
        if options.get("offload") and len(self.act_to_keep) > 0:
            # Free memory just before allocating the next buffer
            activations_offloading().wait_for_offloading()
            
        if src is None or src == self.rank: return
        
        buffers = {}
        for key in self.inputs:
            buffers[key] = [None, self.metadata[key].get_buffer(mb_size)]

        if options.get("batch"):
            # This communication needs to be batched ;
            # instead of executing it, we instanciate an object with the right setup and return it
            self.inputs_to_forward.append(buffers)
            return [dist.P2POp(dist.irecv, buffers[key][1], src) for key in self.inputs]
        
        else:
            logger.debug(f'{self} - Starting to receive activations with shape {self.metadata} from layer {self.id - 1} on rank {src}')
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for key in self.inputs:
                    work = dist.irecv(buffers[key][1], src)
                    buffers[key][0] = work

            torch.cuda.current_stream().wait_stream(stream) # needed ?
            self.inputs_to_forward.append(buffers)

    def recv_backward(self, mb_size, **options):
        '''
        Receive and store one gradient to backward

        :param mb_size: size of the micro batch to receive
        :type mb_size: int
        :param **options: options to modify the send behaviour
        :return: If the communications needs to be batched, returns them
        :rtype: List[dist.P2POp] or None
        '''
        src = options.get("src") or self.next

        if options.get("offload"):
            # Start moving activations back to gpu
            activations_offloading().prefetch()

        if src is None or src == self.rank: return

        buffers = {}
        for key in self.outputs:
            buffers[key] = [None, self.out_metadata[key].get_buffer(mb_size)]

        if options.get("batch"):
            # This communication needs to be batched ;
            # instead of executing it, we instanciate an object with the right setup and return it
            self.grads_to_backward.append(buffers)
            return [dist.P2POp(dist.irecv, buffers[key][1], src) for key in self.outputs]
        
        else:
            logger.debug(f'{self} - Starting to receive gradients with shape {self.out_metadata} from layer {self.id + 1} on rank {src}')
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for key in self.outputs:
                    work = dist.irecv(buffers[key][1], src)
                    buffers[key][0] = work

            torch.cuda.current_stream().wait_stream(stream) # needed ?
            self.grads_to_backward.append(buffers)
    
    def register_metadata(self):
        '''
        Performs a pseudo forward pass to register the input and output shapes

        .. warning::
            We assume that every micro batch has the same shape, except for the micro batch size
        '''
        # Receive metadata of inputs
        if self.previous is not None and self.previous != self.rank:
            for key in self.inputs:
                metadata = torch.empty(TensorMetadata.MAX_SIZE, device = self.device)
                dist.recv(metadata, src = self.previous)
                self.metadata[key] = TensorMetadata.from_tensor(metadata)
            
        # We perform a forward pass with dummy data to get the output shapes (except for batch size) of all blocks
        dummy = {key: self.metadata[key].get_buffer(1) for key in self.inputs}
        for buffer in dummy.values():
            if buffer.dtype == torch.int64:
                buffer[:] = 1 # avoid problems with embeddings that can go out of vocab size :)
        y = self.model(**dummy)

        for key in self.outputs:
            if not isinstance(y[key], torch.Tensor):
                raise RuntimeError(f"Non-tensor output from block {self} : key {key} has type {type(y[key])}.")
            
            # save shapes except batch size
            self.out_metadata[key] = TensorMetadata(y[key].squeeze(0))

        logger.debug(f'{self} - Registered metadata {self.metadata} => {self.out_metadata}')
        
        # Send metadata to next block
        if self.next is not None and self.next != self.rank:
            for key in self.outputs:
                dist.send(self.out_metadata[key].to_tensor(), dst = self.next)

class Pipeline():
    '''
    Model wrapper for pipelining that manages the pipeline setup
    '''
    def __init__(self, model, sample, placement = "auto", partition = "metis", schedule = "afab"):
        '''
        :param model: the entire model to pipeline
        :type model: nn.Module
        :param placement: list of device ranks. Block i of the pipeline will be placed on rank placement[i]. Leave to default ("auto") for automatic placement, which is [0, 1, .., world size - 1]
        :type placement: List[int] or str
        :param partition: if your model is already partitioned, set to False. Otherwise set to the partition strategy you want to use (default = metis), which will try to create balanced blocks according to their number of parameters
        :type partition: boolean or str
        :param schedule: pipeline algorithm to use. currently supported : GPipe ("afab") (default), PipeDream ("1f1b"), Hanayo ("hanayo"). You can also define your own function to generate the schedule, see the existing functions in schedule for an example.
        :type schedule: str or function(List[int], int, **kwargs) -> List[Operation]
        '''
        if not dist.is_initialized() or "RANK" not in os.environ.keys():
            logger.warning(f'Trying to create a pipeline but no multi-gpu distributed setup has been found.')
            
        rank = dist.get_rank()
        if placement == "auto":
            placement = list(range(int(os.environ["WORLD_SIZE"])))
        if isinstance(partition, str):
            model, inputs, outputs = shared_partition(model, placement, sample, partition)
        elif not partition:
            model, inputs, outputs = local_partition(model, placement)
        else:
            raise Exception(f"Partition strategy should be either False for pre-partitioned model, or a string among [naive, constrained, dagP, metis].")

        match schedule.lower():
            case 'afab':
                self.scheduler = generate_afab_schedule
            case '1f1b':
                self.scheduler = generate_1f1b_schedule
            case 'hanayo':
                self.scheduler = generate_hanayo_schedule
            case _:
                self.scheduler = schedule
        
        self.blocks = create_pipeline(model, placement, inputs, outputs)
        self.placement = placement
        self.engine = Engine(self.blocks)
        self.schedule = []
        self.options = {}

    def __call__(self, batch, target, loss_fn, split_size = 1, profile = False, **options):
        '''
        Execute the schedule on a batch of data

        :param split_size: either one size for equal micro batches (last one may be smaller if the batch size is not divisible by the split size), or a list of possibly different micro batch sizes. In that case the sum of the sizes must be equal to the batch size.
        :type split_size: int or List[int]
        :param profile: Whether to activate nvidia profiling or not. If True, NVTX ranges will be generated for each operation
        :type profile: boolean

        :return: result of the forward pass and loss value if the last block of the pipeline is managed by this process
        :rtype: Tensor, Tensor or None, None
        '''
        # Split size can be an int or list of ints ; make it always a list
        if isinstance(split_size, int):
            n_micro_batches = batch.size(0) // split_size
            mb_sizes = [split_size for _ in range(n_micro_batches)]
            if batch.size(0) % split_size != 0:
                mb_sizes.append(batch.size(0) % split_size)
        else:
            assert sum(split_size) == batch.size(0), f'Splits do not cover the entire batch'
            mb_sizes = split_size
            n_micro_batches = len(split_size)
            
        # TODO: the schedule can have a slightly different length with all_reduce etc
        if len(self.schedule) != (n_micro_batches * len(self.blocks) * 3 * 2) or options != self.options:
            # Different number of micro batches ; we have to recompute the schedule
            self.schedule = self.scheduler(self.placement, n_micro_batches, **options)
            self.options = options

            # Construct graph, detech cycle and add communication batching to fix them
            cycles = find_cycles(graph_from_schedule(self.schedule))
            if cycles and dist.get_rank() == 0:
                logger.warning(f'Found potential deadlocks in the schedule ! Fixing them.')
                for c in cycles: logger.debug(f'Cycle : {c}')
            for c in cycles:
                fix_cycle(c)

            # Remove all operations that are not ours
            ids = list(map(lambda b: b.id, self.blocks)) # funny python tips: a map in itself can be iterated only once ! never forget to create a list from it before anything else
            self.schedule = list(filter(lambda op: op.block_id in ids, self.schedule))

        # We expect a list of arguments, not a single tensor
        if isinstance(batch, torch.Tensor):
            batch = [batch]
            
        # Full forward pass to register metadata used to allocate tensors later
        if self.blocks[0].previous is None:
            # Take all
            for k,v in zip(self.blocks[0].inputs, batch):
                self.blocks[0].metadata[k] = TensorMetadata(v[0]) # Don't register batch size

        for i in range(len(self.blocks)):
            b = self.blocks[i]
            # Sync metadata of fused blocks
            if i > 0 and self.blocks[i - 1].rank == b.rank:
                b.metadata = self.blocks[i - 1].out_metadata

            b.register_metadata()
            logger.debug(f'{b} - Registered metadata {b.metadata} -> {b.out_metadata}')

        # Execute the schedule
        result, losses, times = self.engine.train_step(batch, target, loss_fn, self.schedule, mb_sizes, profile)

        self.times = times
        # Merge back the micro-batches outputs/losses into one batch
        if len(result) != 0:
            return torch.cat(result, dim = 0), torch.tensor(losses, device = torch.cuda.current_device()).sum(dim = 0, keepdim = True)
        
        else: return None, None

    def parameters(self):
        return [{
            'params': block.model.parameters()
        } for block in self.blocks]

def create_pipeline(layers, placement, inputs, outputs):
    '''
    Transforms a list of layers placed on different devices to a working pipeline

    :param layers: List of layers / groups of layers coming from a partitioned model. This list has to be sequential in terms of computation
    :type layers: List[nn.Module]
    :param placement: list of device ranks
    :type placement: List[int]
    :param inputs: name of variables taken as input by each block
    :type inputs: List[List[str]]
    :param outputs: name of variables returned by each block
    :type outputs: List[List[str]]

    :return: list of blocks handled by this process with everything set up for the pipeline to work
    :rtype: List[PipelineBlock]
    '''
    rank = int(os.getenv("RANK")) if "RANK" in os.environ.keys() else 'cpu'

    ids = [i for i in range(len(placement)) if placement[i] == rank]
    blocks = []
    for i in range(len(layers)):
        new_block = PipelineBlock(layers[i], ids[i], placement, inputs[i], outputs[i])
        blocks.append(new_block)
        logger.info(f'{new_block} : inputs = {new_block.inputs}, outputs = {new_block.outputs}')

    return blocks

def shared_partition(model, placement, sample, mode):
    '''
    Partitions a model according to a placement & mode, then shares it to every process to be consistent

    :param model: model to partition
    :type model: nn.Module
    :param placement: list of device ranks
    :type placement: List[int]
    :param sample: example of input data that will be processed by the model
    :type sample: Tensor
    :param mode: partitioner to use ; available options are :
    
        - "naive": simple load balancing algorithm
        - "constrained": naive with less communication
        - "metis": use METIS
        - "dagP": use dagP / rMLGP
        For more info, see partition.

    :type mode: str

    :return:

        - Blocks for this process
        - Inputs for each block of this process
        - Outputs for each block of this process 

    :rtype: List[nn.Module], List[List[str]], List[List[str]]
    '''
    rank = dist.get_rank()
    
    # Rank 0 profiles & partition the graph, then shares it to everyone
    # TODO: what if devices are heterogenous ? how to profile correctly ?
    input_list = None
    if rank == 0:
        assert mode in ["naive", "constrained", "dagP", "metis"], "Partition strategies available are : [naive, constrained, dagP, metis]"

        blocks, inputs, outputs = partition_graph(model, len(placement), sample, mode = mode)
        partition = list(zip(blocks, inputs.values(), outputs.values()))
        input_list = [[] for _ in range(max(placement) + 1)]
        for i, p in enumerate(placement):
            input_list[p].append(partition[i])

    output_list = [None]
    dist.scatter_object_list(output_list, input_list, src = 0)
    model, inputs, outputs = ([m.cuda() for m, _, _ in output_list[0]],
                              [i for _, i, _ in output_list[0]],
                              [o for _, _, o in output_list[0]])
    return model, inputs, outputs

def local_partition(model, placement):
    '''
    Partitions a pre-partitioned model locally for each process.

    This function takes a model that has already been partitioned and identifies the model parts
    assigned to the current rank, traces them, and extracts their inputs and outputs.

    :param model: The pre-partitioned model. Can be a single nn.Module or a list of nn.Modules.
    :type model: nn.Module or List[nn.Module]
    :param placement: A list indicating the rank assignment for each part of the model.
    :type placement: List[int]

    :return: A tuple containing:
        - The partitioned model parts for the current rank
        - A list of inputs for each model part
        - A list of outputs for each model part
    :rtype: Tuple[List[nn.Module], List[List[str]], List[List[str]]]
    '''
    rank = dist.get_rank()
    inputs = []
    outputs = []
    if isinstance(model, torch.nn.Module): model = [model]
    ids = [i for i in range(len(placement)) if placement[i] == rank]
    for i in range(len(ids)):
        trace = torch.fx.symbolic_trace(model[i])
        new, inputs[i], outputs[i] = get_inputs_outputs_single(list(trace.graph.nodes))
        model[i] = create_subgraph(trace, new, inputs[i], outputs[i])

    return model, inputs, outputs