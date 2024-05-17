import os
import torch
import torch.nn as nn
import torch.distributed as dist
from .schedule import *
from .engine import Engine
from collections import deque
import logging
logger = logging.getLogger("pipeline")

dtypes = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool
]

class TensorMetadata():
    '''
    Informations about Tensors that are sent and received in p2p communication
    [dtype, *shape]
    '''
    MAX_SIZE = 16

    @staticmethod
    def from_tensor(t):
        '''
        Creates a TensorMetadata object from its Tensor equivalent (should be used when receiving metadata via p2p)
        '''
        dtype = dtypes[int(t[0].item())]
        shape = []
        assert len(t.shape) == 1, "Metadata should only have one dimension"
        for s in t[1:]:
            s = int(s.item())
            if s == 0: break
            shape.append(s)
        
        metadata = TensorMetadata(torch.empty(0, dtype=dtype))
        metadata.shape = shape
        return metadata

    def __init__(self, t):
        self.shape = t.shape
        self.dtype = t.dtype
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    def to_tensor(self):
        '''
        Creates the Tensor representation of this metadata. Should be used when sending metadata via p2p
        '''
        t = torch.zeros(TensorMetadata.MAX_SIZE, device = self.device)
        t[0] = dtypes.index(self.dtype)
        for i, s in enumerate(self.shape):
            t[1 + i] = s
        return t
    
    def get_buffer(self, batch_size):
        '''
        Allocates a tensor with the right shape and dtype for this metadata
        '''
        buffer = torch.empty((batch_size, *self.shape), dtype = self.dtype, device = self.device)
        return buffer

class PipelineBlock():
    '''
    Manages one layer/group of contiguous layers placed on one device
    '''
    def __init__(self, model, id_, placement):
        super(PipelineBlock, self).__init__()
        # Block infos
        self.rank = placement[id_] # global rank
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.id = id_ # rank in the model.
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

        self.metadata = None
        self.out_metadata = None

    def __str__(self) -> str:
        return f'[Layer {self.id} : GPU {self.rank}]'

    def forward(self, options = {}):
        '''
        Perform the forward pass for one tensor of activations and register it as computed
        '''
        if len(self.inputs) == 0: return

        logger.debug(f'{self} - Computing one forward')
        
        work, x = self.inputs.popleft()

        if work is not None: work.wait()
        
        if x.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            x.requires_grad = True

        if options and ('remat' in options.keys()):
            with torch.no_grad(): y = self.model(x)        
        else:
            logger.debug(f'{self} - Forwarding tensor with shape {x.shape}')
            y = self.model(x)
            self.activations.append(y)
           
        self.act_to_send.append(y)
        self.inputs_to_keep.append(x)

        if self.next is None:
            return self.act_to_send.popleft()
        
    def backward(self, options = {}):
        '''
        Perform the backward pass for one tensor of gradients and register it as computed
        Backward assumes activations AND grads to be on top of the queue
        '''
        if len(self.grads) == 0: return
        logger.debug(f'{self} - Computing one backward')

        x = self.inputs_to_keep.popleft()
        if options and 'remat' in options:
            act = self.model(x)
        else:
            act = self.activations.popleft()
        work, grads = self.grads.popleft()
        
        if work is not None: work.wait()
        
        (act * grads).sum().backward() # Optimal ? Maybe setting the gradients directly is faster

        if x.requires_grad:
            self.grads_to_send.append(x.grad.data)
            
    def send_forward(self, options = {}):
        '''
        Send one activation to the next layer in the model
        '''
        dst = options["dst"] if (options and "dst" in options.keys()) else self.next
        if dst is None or len(self.act_to_send) == 0: return

        activations = self.act_to_send.popleft()

        logger.debug(f'{self} - Sending activations to layer {self.id + 1} on rank {dst}')
        if "batch" in options.keys(): return dist.P2POp(dist.isend, activations, dst)
        else: dist.isend(activations, dst)

    def send_backward(self, options = {}):
        '''
        Send one gradient to the previous layer in the model
        '''
        dst = options["dst"] if options and "dst" in options.keys() else self.previous
        if dst is None or len(self.grads_to_send) == 0: return

        grads = self.grads_to_send.popleft()

        logger.debug(f'{self} - Sending gradients to layer {self.id - 1} on rank {dst}')
        if "batch" in options.keys(): return dist.P2POp(dist.isend, grads, dst)
        else: dist.isend(grads, dst)

    def recv_forward(self, mb_size, options = {}):
        '''
        Receive and store one activation to forward
        '''
        src = options["src"] if options and "src" in options.keys() else self.previous
        if src is None: return

        buffer = self.metadata.get_buffer(mb_size)

        logger.debug(f'{self} - Waiting for activations with shape {buffer.shape} from layer {self.id - 1} on rank {src}')

        if "batch" in options.keys():
            self.inputs.append((None, buffer))
            return dist.P2POp(dist.irecv, buffer, src)
        else:
            work = dist.irecv(buffer, src)
            self.inputs.append((work, buffer)) # .detach() ?

    def recv_backward(self, mb_size, options = {}):
        '''
        Receive and store one gradient to backward
        '''
        src = options["src"] if options and "src" in options.keys() else self.next
        if src is None: return

        buffer = self.out_metadata.get_buffer(mb_size)
        logger.debug(f'{self} - Waiting for gradients with shape {buffer.shape} from layer {self.id + 1} on rank {src}')

        if "batch" in options.keys():
            self.grads.append((None, buffer))
            return dist.P2POp(dist.irecv, buffer, src)
        else:
            work = dist.irecv(buffer, src)
            self.grads.append((work, buffer))
    
    def register_metadata(self):
        if self.previous is not None:
            metadata = torch.empty(TensorMetadata.MAX_SIZE, device = self.device)
            dist.recv(metadata, src = self.previous)
            self.metadata = TensorMetadata.from_tensor(metadata)
            
        dummy = self.metadata.get_buffer(1)
        dummy[:] = 1 # avoid problems with embeddings :)
        y = self.model(dummy)

        out_metadata = TensorMetadata(y.squeeze(0)) # do not include batch size
        self.out_metadata = out_metadata
        
        if self.next is not None:
            dist.send(out_metadata.to_tensor(), dst = self.next)

class Pipeline():
    '''
    Model wrapper for pipelining
    Maybe it can inherit from nn.Module ?
    '''
    def __init__(self, model, placement = "auto", partition = "auto", schedule = "afab"):
        '''
        model: torch module
        placement: ranks on which each model block should be placed
        partition: if your model is already partitioned, set to None. Otherwise leave the default, which will try to create balanced blocks according to their number of parameters
        schedule: pipeline algorithm to use. currently supported : GPipe ("afab"), PipeDream ("1f1b")
        '''
        if not dist.is_initialized() or "RANK" not in os.environ.keys():
            logger.warning(f'Trying to create a pipeline but no multi-gpu distributed setup has been found.')
        if placement == "auto":
            placement = list(range(int(os.environ["WORLD_SIZE"])))
        if partition == "auto":
            model = partition_model(model, placement)
        match schedule.lower():
            case 'afab':
                self.scheduler = generate_afab_schedule
            case '1f1b':
                self.scheduler = generate_1f1b_schedule
            case 'hanayo':
                self.scheduler = generate_hanayo_schedule
            case _:
                raise Exception(f'Unknown schedule : {schedule}. Possible options are ["afab", "1f1b"].')
        
        self.blocks = create_pipeline(model, placement)
        self.placement = placement
        self.engine = Engine(self.blocks)
        self.schedule = []

    def __call__(self, batch, target, loss_fn, split_size = 1, viz_file = None):
        '''
        Execute the schedule on a batch of data
        split_size: either int for equal micro batches (last one may be smaller if the batch size is not divisible by the split size), or list of micro batch sizes
        viz_file: path to a file to write informations about the execution. Set to None (default) to disable
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
            
        if len(self.schedule) != (n_micro_batches * len(self.placement) * 3 * 2): # Different number of micro batches ; we have to recompute the schedule 

            self.schedule = self.scheduler(self.placement, n_micro_batches)
            cycles = find_cycles(graph_from_schedule(self.schedule))
            if cycles and self.blocks[0].rank == 0: logger.warning(f'Found potential deadlocks in the schedule ! Fixing them.')
            for c in cycles:
                fix_cycle(c)
            
        # Full forward pass to register metadata used to allocate tensors later
        if self.blocks[0].previous is None:
            self.blocks[0].metadata = TensorMetadata(batch[0])
        for b in self.blocks:
            b.register_metadata()
            logger.debug(f'{b} - Registered metadata {b.metadata.shape} -> {b.out_metadata.shape}')

        result, losses = self.engine.train_step(batch, target, loss_fn, self.schedule, split_size, viz_file)
        if len(result) != 0:
            return torch.cat(result, dim = 0), torch.tensor(losses, device = torch.cuda.current_device()).sum(dim = 0, keepdim = True)
        else: return None, None

def create_pipeline(layers, placement):
    '''
    Transforms a list of layers placed on different devices to a working pipeline
    '''
    rank = int(os.getenv("RANK")) if "RANK" in os.environ.keys() else 'cpu'
    local_rank = int(os.getenv("LOCAL_RANK")) if "LOCAL_RANK" in os.environ.keys() else 'cpu'

    ids = [i for i in range(len(placement)) if placement[i] == rank]
    blocks = [PipelineBlock(layer, i, placement) for i, layer in zip(ids, layers)]

    # Merge consecutive blocks that are on the same device (useful for hanayo-like schedules)
    i = 0
    while i < len(blocks) - 1:
        if blocks[i + 1].id == blocks[i].id + 1:
            merged_block = PipelineBlock(nn.Sequential(blocks[i].model, blocks[i + 1].model), blocks[i].id, placement)
            merged_block.previous = blocks[i].previous
            merged_block.next = blocks[i + 1].next
            blocks[i] = merged_block
            blocks.pop(i + 1)
            i -= 1
        i += 1

    return blocks

def partition_model(model, placement):
    '''
    Divide a model into roughly equal blocks, in terms of number of parameters
    Each block that is placed on this rank is moved to the corresponding GPU
    '''
    rank = dist.get_rank() if dist.is_initialized() else None

    available_devices = [torch.device(device) for device in ['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]]

    if any(torch.device(p) not in available_devices for p in placement):
        raise RuntimeError(f'Trying to place the model on non existing or non visible devices : {placement}, but available devices are : {available_devices}')
    
    layers = []
    for module in model.children():
        if isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential):
            layers.extend(module)
        else:
            layers.append(module)
            
    def numel(layer):
        return sum([p.numel() for p in layer.parameters()])

    total_numel = sum([numel(layer) for layer in layers])
    phase_numel = total_numel // len(placement)
    delim_numel = phase_numel
    accum_numel = 0

    # seal one pipeline phase when its numel is larger than phase_numel
    phases = [[]]
    for layer in layers:
        phases[-1].append(layer)
        accum_numel += numel(layer)
        if accum_numel >= delim_numel:
            delim_numel += phase_numel
            phases.append([])

    # pack all remaining layers into the last phase
    while len(phases) > len(placement):
        phases[-2].extend(phases[-1])
        phases.pop()

    if rank is None:
        for i, phase in enumerate(phases):
            for layer in phase:
                layer.to_empty(device = torch.device(placement[i]))
    else:
        for i in range(len(placement)):
            if rank == placement[i]:
                for layer in phases[i]:
                    layer.to_empty(device=torch.device(rank))

    # create nn.Sequential
    if rank is None: return nn.Sequential(*[nn.Sequential(*phase) for phase in phases])
    else: return [nn.Sequential(*phases[i]) for i in range(len(placement)) if placement[i] == rank]
