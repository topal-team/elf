'''
Core pipeline objects. Define the user interface (Pipeline) and the behaviour of individual stages (PipelineBlock), by managing the p2p communications and data queues.
'''

import os
import torch
import shutil
import torch.distributed as dist
from .schedule import *
from .engine import Engine
from .utils import Timer, TensorMetadata, activations_offloading
from .partitioners import partition_graph, get_inputs_outputs_single
from collections import deque
import logging
logger = logging.getLogger("pipeline")

class PipelineBlock():
    '''
    Manages one layer/group of contiguous layers placed on one device
    '''
    def __init__(self, model, id_, placement, params, outputs):
        super(PipelineBlock, self).__init__()
        # Block infos
        self.rank = placement[id_] # global rank
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.id = id_ # rank in the model
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

        # Queues of tensors to process
        self.inputs = deque() # Waiting for forward
        self.activations = deque() # Kept for backward
        self.grads = deque() # Waiting for backward
        self.act_to_send = deque() # Sent to next block
        self.grads_to_send = deque() # Sent to previous block
        self.inputs_to_keep = deque() # Kept for backward

        # Ranks where the previous/next blocks in the model are placed
        self.previous = None if self.id == 0 else placement[self.id - 1]
        self.next = None if self.id == len(placement) - 1 else placement[self.id + 1]

        self.metadata = {}
        self.out_metadata = {}

        self.compute_time = 0 # used to measure idle time

        self.params = sorted(params) # name of input variables
        self.outputs = sorted(outputs) # name of output variables
        # sorted alphabetically to make sure the order is consistent across devices

    def __str__(self) -> str:
        return f'[Layer {self.id} : GPU {self.rank}]'

    def forward(self, **options):
        '''
        Perform the forward pass for one tensor of activations and register it as computed
        '''
        logger.debug(f'{self} - Computing one forward with options {options}')
        
        x = self.inputs.popleft()
        for key in self.params:
            work, i = x[key]
            if work is not None: work.wait()
            x[key] = i
        
            if x[key].dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                x[key].requires_grad = True

        # logger.debug(f'{self} - Forwarding tensor with shape {x.shape}')

        with Timer() as timer:
            if  options.get('remat'):
                with torch.no_grad(): y = self.model(**x)
            elif options.get('offload'):
                with activations_offloading():
                    y = self.model(**x)
                self.activations.append(y)
                # x = x.cpu()
            else:
                y = self.model(**x)
                self.activations.append(y)
                
        self.compute_time += timer.time()
           
        self.act_to_send.append(y)
        self.inputs_to_keep.append(x)

        if self.next is None:
            return self.act_to_send.popleft()
        
    def backward(self, **options):
        '''
        Perform the backward pass for one tensor of gradients and register it as computed
        Backward assumes activations AND grads to be on top of the queue
        '''
        logger.debug(f'{self} - Computing one backward with options {options}')

        x = self.inputs_to_keep.popleft()

        if options.get('remat'):
            with Timer() as timer:
                act = self.model(**x)
            self.compute_time += timer.time()
        elif options.get('offload'):
            act = self.activations.popleft().cuda()
            # x = x.cuda()
        else:
            act = self.activations.popleft()
            
        grads = self.grads.popleft()
        for key in self.outputs:
            work, g = grads[key]
            if work is not None: work.wait()
            grads[key] = g

        with Timer() as timer:
            for key in self.outputs:
                act[key].backward(grads[key], retain_graph = (key != self.outputs[-1]))
        self.compute_time += timer.time()

        self.grads_to_send.append({key: value.grad.data for key, value in x.items() if value.requires_grad})
            
    def send_forward(self, **options):
        '''
        Send one activation to the next layer in the model
        If the communication needs to be batched, returns it as a dist.P2POp. Otherwise returns None.
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
        Send one gradient to the previous layer in the model
        If the communication needs to be batched, returns it as a dist.P2POp. Otherwise returns None.
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
        If the communication needs to be batched, returns it as a dist.P2POp. Otherwise returns None.
        '''
        src = options.get("src") or self.previous
        if options.get("offload") and len(self.activations) > 0:
            # Free memory just before allocating the next buffer
            activations_offloading().wait_for_offloading()
            
        if src is None or src == self.rank: return
        
        buffers = {key: [None, self.metadata[key].get_buffer(mb_size)] for key in self.params}

        if options.get("batch"):
            self.inputs.append(buffers)
            return [dist.P2POp(dist.irecv, buffers[key][1], src) for key in self.params]
        else:
            # logger.debug(f'{self} - Starting to receive activations with shape {buffer.shape} from layer {self.id - 1} on rank {src}')
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for key in self.params:
                    work = dist.irecv(buffers[key][1], src)
                    buffers[key][0] = work
            torch.cuda.current_stream().wait_stream(stream)
            self.inputs.append(buffers)

    def recv_backward(self, mb_size, **options):
        '''
        Receive and store one gradient to backward
        If the communication needs to be batched, returns it as a dist.P2POp. Otherwise returns None.
        '''
        src = options.get("src") or self.next

        if options.get("offload"):
            # Start moving activations back to gpu
            activations_offloading().prefetch()

        if src is None or src == self.rank: return

        buffers = {key: [None, self.out_metadata[key].get_buffer(mb_size)] for key in self.outputs}

        if options.get("batch"):
            self.grads.append(buffers)
            return [dist.P2POp(dist.irecv, buffers[key][1], src) for key in self.outputs]
        else:
            # logger.debug(f'{self} - Starting to receive gradients with shape {buffer.shape} from layer {self.id + 1} on rank {src}')
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for key in self.outputs:
                    work = dist.irecv(buffers[key][1], src)
                    buffers[key][0] = work
            torch.cuda.current_stream().wait_stream(stream)
            self.grads.append(buffers)
    
    def register_metadata(self):
        '''
        Performs a pseudo forward pass to register the input and output shapes
        !! We assume that every micro batch has the same shape, except for the micro batch size !!
        '''
        if self.previous is not None and self.previous != self.rank:
            for key in self.params:
                metadata = torch.empty(TensorMetadata.MAX_SIZE, device = self.device)
                dist.recv(metadata, src = self.previous)
                self.metadata[key] = TensorMetadata.from_tensor(metadata)
            
        dummy = {key: self.metadata[key].get_buffer(1) for key in self.params}
        if 'idx' in dummy:
            dummy['idx'][:] = 1 # avoid problems with embeddings :)
        y = self.model(**dummy)
        
        self.out_metadata = {key: TensorMetadata(y[key].squeeze(0)) for key in self.outputs} # do not include batch size
        
        if self.next is not None and self.next != self.rank:
            for key in self.outputs:
                dist.send(self.out_metadata[key].to_tensor(), dst = self.next)

class Pipeline():
    '''
    Model wrapper for pipelining
    Maybe it can inherit from nn.Module ?
    '''
    def __init__(self, model, sample, placement = "auto", partition = "auto", schedule = "afab"):
        '''
        model: torch module
        placement: list of device ranks. Block i of the pipeline will be placed on rank placement[i]. Leave to default ("auto") for automatic placement, which is [0, 1, .., world size - 1]
        partition: if your model is already partitioned, set to None. Otherwise leave the default ("auto"), which will try to create balanced blocks according to their number of parameters
        schedule: pipeline algorithm to use. currently supported : GPipe ("afab") (default), PipeDream ("1f1b"), Hanayo ("hanayo")
        '''
        if not dist.is_initialized() or "RANK" not in os.environ.keys():
            logger.warning(f'Trying to create a pipeline but no multi-gpu distributed setup has been found.')
            
        rank = dist.get_rank()
        if placement == "auto":
            placement = list(range(int(os.environ["WORLD_SIZE"])))
        if partition == "auto":
            if shutil.which("gpmetis"):
                if rank == 0: logger.info(f'Using METIS to partition the graph.')
                mode = "metis"
            else:
                if rank == 0: logger.info(f'METIS not found; relying on manual graph partitioning. Consider installing METIS as it is more efficient: https://github.com/KarypisLab/METIS')
                mode = "default"
            model, inputs, outputs = share_partition(model, placement, sample, mode)
        else:
            inputs, outputs = get_inputs_outputs_single(torch.fx.symbolic_trace(model))

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
        split_size: either int for equal micro batches (last one may be smaller if the batch size is not divisible by the split size), or list of micro batch sizes
        profile: Whether to activate nvidia profiling or not. If True, NVTX ranges will be generated for each operation
        '''
        if isinstance(split_size, int):
            n_micro_batches = batch.size(0) // split_size
            s = [split_size for _ in range(n_micro_batches)]
            if batch.size(0) % split_size != 0:
                s.append(batch.size(0) % split_size)
            split_size = s
        else:
            assert sum(split_size) == batch.size(0), f'Splits do not cover the entire batch'
            n_micro_batches = len(split_size)
            
        if len(self.schedule) != (n_micro_batches * len(self.blocks) * 3 * 2) or options != self.options: # Different number of micro batches ; we have to recompute the schedule 

            self.schedule = self.scheduler(self.placement, n_micro_batches, **options)
            self.options = options
            cycles = find_cycles(graph_from_schedule(self.schedule))
            if cycles and dist.get_rank() == 0:
                logger.warning(f'Found potential deadlocks in the schedule ! Fixing them.')
                for c in cycles: logger.debug(f'Cycle : {c}')
            for c in cycles:
                fix_cycle(c)
            # Remove all operations that are not ours
            ids = list(map(lambda b: b.id, self.blocks)) # funny python tips: a map in itself can be iterated only once ! never forget to create a list from it before anything else
            self.schedule = list(filter(lambda op: op.block_id in ids, self.schedule))
            
        # Full forward pass to register metadata used to allocate tensors later
        if self.blocks[0].previous is None:
            self.blocks[0].metadata = {'idx': TensorMetadata(batch[0])} # TODO: should accept multiple inputs
        for i in range(len(self.blocks)):
            b = self.blocks[i]
            if i > 0 and self.blocks[i - 1].rank == b.rank:
                b.metadata = self.blocks[i - 1].out_metadata
            b.register_metadata()
            # logger.debug(f'{b} - Registered metadata {b.metadata.shape} -> {b.out_metadata.shape}')

        result, losses, times = self.engine.train_step(batch, target, loss_fn, self.schedule, split_size, profile)
        self.times = times
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
    '''
    rank = int(os.getenv("RANK")) if "RANK" in os.environ.keys() else 'cpu'

    ids = [i for i in range(len(placement)) if placement[i] == rank]
    blocks = [PipelineBlock(layer, idx, placement, inputs[i], outputs[i]) for i, (idx, layer) in enumerate(list(zip(ids, layers)))]
    for b in blocks:
        logger.info(f'{b} : inputs = {b.params}, outputs = {b.outputs}')

    return blocks

def share_partition(model, placement, sample, mode):
    '''
    Partitions a model according to a placement & mode, then shares it to every process to be consistent
    '''
    rank = dist.get_rank()
    # Rank 0 profiles & partition the graph, then shares it to everyone
    # TODO: avoid loading everything on rank 0 as it can easily OOM
    # TODO: what if devices are heterogenous ? how to profile correctly ?
    if rank == 0:
        blocks, inputs, outputs = partition_graph(model, len(placement), sample, mode = mode)
        partition = list(zip(blocks, inputs.values(), outputs.values()))
        input_list = [[] for _ in range(max(placement) + 1)]
        for i, p in enumerate(placement):
            input_list[p].append(partition[i])
    else:
        input_list = None
    output_list = [None]
    dist.scatter_object_list(output_list, input_list, src = 0)
    model, inputs, outputs = ([m for m, _, _ in output_list[0]],
                              [i for _, i, _ in output_list[0]],
                              [o for _, _, o in output_list[0]])
    return model, inputs, outputs