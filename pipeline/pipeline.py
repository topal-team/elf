from typing import Any
import torch
import torch.nn as nn
import torch.distributed as dist
from .schedule import generate_afab_schedule, generate_1f1b_schedule
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
    MAX_SIZE = 64

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

    def to_tensor(self):
        '''
        Creates the Tensor representation of this metadata. Should be used when sending metadata via p2p
        '''
        t = torch.zeros(TensorMetadata.MAX_SIZE).cuda()
        t[0] = dtypes.index(self.dtype)
        for i, s in enumerate(self.shape):
            t[1 + i] = s
        return t
    
    def get_buffer(self):
        '''
        Allocates a tensor with the right shape for this metadata
        '''
        buffer = torch.empty(self.shape, dtype=self.dtype).cuda()
        return buffer

class PipelineBlock():
    '''
    Manages one layer/group of contiguous layers placed on one device
    '''
    def __init__(self, model, id_, placement):
        super(PipelineBlock, self).__init__()
        # Block infos
        self.model = model.cuda()
        self.rank = placement[id_] # global rank
        self.id = id_ # rank in the model.

        # Queues of tensor to process
        self.inputs = deque() # Waiting for forward
        self.activations = deque() # Kept for backward
        self.grads = deque() # Waiting for backward
        self.act_to_send = deque() # Sent to next block
        self.grads_to_send = deque() # Sent to previous block
        self.inputs_to_keep = deque() # Kept for backward

        # Ranks where the previous/next blocks in the model are placed
        # OR reference to the next block if they are on the same device
        self.previous = None if self.id == 0 else placement[self.id - 1]
        self.next = None if self.id == len(placement) - 1 else placement[self.id + 1]

    def __str__(self) -> str:
        return f'[Layer {self.id} : GPU {self.rank}]'

    def forward(self):
        '''
        Perform the forward pass for one tensor of activations and register it as computed
        '''
        if len(self.inputs) == 0: return

        logger.debug(f'{self} - Computing one forward')
        
        work, x = self.inputs.popleft()

        if work is not None: work.wait() # if properly managed, work should already be completed
        
        if x.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            x.requires_grad = True
            
        logger.debug(f'{self} - Work received. Starting actual computation.')
        y = self.model(x)
        self.activations.append(y)
        self.act_to_send.append(y)
        self.inputs_to_keep.append(x)

        if self.next is None:
            return self.act_to_send.popleft()
        
    def backward(self):
        '''
        Perform the backward pass for one tensor of gradients and register it as computed
        Backward assumes activations AND grads to be on top of the queue
        '''
        if len(self.grads) == 0: return
        logger.debug(f'{self} - Computing one backward')

        act = self.activations.popleft()
        work, grads = self.grads.popleft()
        
        if work is not None: work.wait() # if properly managed, work should alredy be completed
        
        x = self.inputs_to_keep.popleft()
        (act * grads).sum().backward() # Optimal ? Maybe setting the gradients directly is faster

        if x.requires_grad:
            self.grads_to_send.append(x.grad.data)
            
    def send_forward(self):
        '''
        Send one activation to the next layer in the model
        '''
        if self.next is None or len(self.act_to_send) == 0: return

        activations = self.act_to_send.popleft()

        metadata = TensorMetadata(activations).to_tensor()
        dist.isend(metadata, self.next)

        logger.debug(f'{self} - Sending activations to layer {self.id + 1} on rank {self.next}')
        dist.isend(activations, self.next)

    def send_backward(self):
        '''
        Send one gradient to the previous layer in the model
        '''
        if self.previous is None or len(self.grads_to_send) == 0: return

        grads = self.grads_to_send.popleft()

        metadata = TensorMetadata(grads).to_tensor()
        dist.isend(metadata, self.previous)

        logger.debug(f'{self} - Sending gradients to layer {self.id - 1} on rank {self.previous}')
        dist.isend(grads, self.previous)

    def recv_forward(self):
        '''
        Receive and store one activation to forward
        '''
        if self.previous is None: return

        buffer = torch.empty(TensorMetadata.MAX_SIZE).cuda()
        work = dist.irecv(buffer, self.previous)
        logger.debug(f'{self} - Waiting for metadata of activations from rank {self.previous}')
        work.wait()
        metadata = TensorMetadata.from_tensor(buffer)
        buffer = metadata.get_buffer()

        logger.debug(f'{self} - Waiting for activations with shape {buffer.shape} from layer {self.id - 1} on rank {self.previous}')
        work = dist.irecv(buffer, self.previous)
        logger.debug(f'{self} - Received activations !')

        self.inputs.append((work, buffer)) # .detach() ?

    def recv_backward(self):
        '''
        Receive and store one gradient to backward
        '''
        if self.next is None: return

        buffer = torch.empty(TensorMetadata.MAX_SIZE).cuda()

        work = dist.irecv(buffer, self.next)
        logger.debug(f'{self} - Waiting for metadata for gradients from layer {self.id + 1} on rank {self.next}')
        work.wait()

        metadata = TensorMetadata.from_tensor(buffer)
        buffer = metadata.get_buffer()

        logger.debug(f'{self} - Waiting for gradients with shape {buffer.shape} from layer {self.id + 1} on rank {self.next}')
        work = dist.irecv(buffer, self.next)
        logger.debug(f'{self} - Received gradients !')

        self.grads.append((work, buffer))

class Pipeline():
    def __init__(self, model, placement, partition="auto", schedule="afab"):
        if partition == "auto":
            model = partition_model(model, placement)
        match schedule:
            case 'afab':
                self.scheduler = generate_afab_schedule
            case '1f1b':
                self.scheduler = generate_1f1b_schedule
            case _:
                raise Exception(f'Unknown schedule : {schedule}. Possible options are ["afab", "1f1b"].')
        
        self.blocks = create_pipeline(model, placement)
        self.placement = placement
        self.__call__ = self.forward

    def forward(self, batch, target, loss_fn, split_size = 1, viz_file = None):
        if isinstance(split_size, int):
            n_micro_batches = batch.size(0) // split_size
        else:
            n_micro_batches = len(split_size)
        schedule = self.scheduler(self.placement, n_micro_batches)

        engine = Engine(schedule, self.blocks)
        result = engine.train_step(batch, target, loss_fn, split_size, viz_file)
        if result: return torch.cat(result, dim=0)
        
def create_pipeline(layers, placement):
    '''
    Transforms a list of layers placed on different devices to a working pipeline
    '''
    rank = dist.get_rank()

    ids = [i for i in range(len(placement)) if placement[i] == rank]
    blocks = [PipelineBlock(layer, i, placement) for i, layer in zip(ids, layers)]
    return blocks

def partition_model(model, placement):
    '''
    Divide a model into roughly equal blocks, in terms of number of parameters
    Each block that is placed on this rank is moved to the corresponding GPU
    Currently does not support interleaving / multiple blocks per device
    '''
    rank = dist.get_rank() if dist.is_initialized() else None
    
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
        if accum_numel > delim_numel:
            delim_numel += phase_numel
            phases.append([])

    # pack all remaining layers into the last phase
    while len(phases) > len(placement):
        phases[-2].extend(phases[-1])
        phases.pop()

    if rank is None:
        for i, phase in enumerate(phases):
            for layer in phase:
                layer.to_empty(device=torch.device(placement[i]))
    else:
        for i in range(len(placement)):
            if rank == placement[i]:
                for layer in phases[i]:
                    layer.to_empty(device=torch.device(rank))

    # create nn.Sequential
    if rank is None: return nn.Sequential(*[nn.Sequential(*phase) for phase in phases])
    else: return [nn.Sequential(*phases[i]) for i in range(len(placement)) if placement[i] == rank]
