import torch
import torch.distributed as dist
from collections import deque
import logging
logger = logging.getLogger("pipeline")

class TensorMetadata():
    '''
    Informations about Tensors that are sent and received in p2p communication
    '''
    MAX_SIZE = 64

    @staticmethod
    def from_tensor(t):
        '''
        Creates a TensorMetadata object from its Tensor equivalent (should be used when receiving metadata via p2p)
        '''
        shape = []
        assert len(t.shape) == 1, "Metadata should only have one dimension"
        for s in t:
            s = int(s.item())
            if s == 0: break
            shape.append(s)
        
        metadata = TensorMetadata(torch.empty(0))
        metadata.shape = shape
        return metadata

    def __init__(self, t):
        self.shape = t.shape

    def to_tensor(self):
        '''
        Creates the Tensor representation of this metadata. Should be used when sending metadata via p2p
        '''
        t = torch.zeros(TensorMetadata.MAX_SIZE).cuda()
        for i, s in enumerate(self.shape):
            t[i] = s
        return t
    
    def get_buffer(self):
        '''
        Allocates a tensor with the right shape for this metadata
        '''
        buffer = torch.empty(self.shape).cuda()
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
        logger.debug(f'{self} - Computing forward')
        work, x = self.inputs.popleft()
        if work is not None: work.wait() # if properly managed, work should alredy be completed
        x.requires_grad = True
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
        logger.debug(f'{self} - Computing backward')

        act = self.activations.popleft()
        work, grads = self.grads.popleft()
        if work is not None: work.wait() # if properly managed, work should alredy be completed
        x = self.inputs_to_keep.popleft()

        (act * grads).sum().backward()

        self.grads_to_send.append(x.grad.data)
        if self.previous is None:
            return self.grads_to_send.popleft()

    def send_forward(self):
        '''
        Send one activation to the next layer in the model
        '''
        if self.next is None or len(self.act_to_send) == 0: return

        activations = self.act_to_send.popleft()

        metadata = TensorMetadata(activations).to_tensor()
        dist.send(metadata, self.next)

        logger.debug(f'{self} - Sending activations to layer {self.id + 1} on rank {self.next}')
        dist.isend(activations, self.next)


    def send_backward(self):
        '''
        Send one gradient to the previous layer in the model
        '''
        if self.previous is None or len(self.grads_to_send) == 0: return

        grads = self.grads_to_send.popleft()

        metadata = TensorMetadata(grads).to_tensor()
        dist.send(metadata, self.previous)

        logger.debug(f'{self} - Sending gradients to layer {self.id - 1} on rank {self.previous}')
        dist.isend(grads, self.previous)


    def recv_forward(self):
        '''
        Receive and store one activation to forward
        '''
        if self.previous is None or isinstance(self.previous, PipelineBlock): return

        buffer = torch.empty(TensorMetadata.MAX_SIZE).cuda()
        dist.recv(buffer, self.previous) # TODO : async here ?
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
        if self.next is None or isinstance(self.next, PipelineBlock): return

        buffer = torch.empty(TensorMetadata.MAX_SIZE).cuda()
        dist.recv(buffer, self.next) # TODO : async here ?
        metadata = TensorMetadata.from_tensor(buffer)
        buffer = metadata.get_buffer()

        logger.debug(f'{self} - Waiting for gradients with shape {buffer.shape} from layer {self.id + 1} on rank {self.next}')
        work = dist.irecv(buffer, self.next)
        logger.debug(f'{self} - Received gradients !')

        self.grads.append((work, buffer))

    def has_work_forward(self):
        for (work, _) in self.inputs:
            if work is None or work.is_completed():
                return True
        return False

    def has_work_backward(self):
        for (work, _) in self.grads:
            if work is None or work.is_completed():
                return True
        return False

def create_pipeline(layers, placement):
    '''
    Transforms a list of layers placed on different devices to a working pipeline
    '''
    rank = dist.get_rank()

    ids = [i for i in range(len(placement)) if placement[i] == rank]
    blocks = [PipelineBlock(layer, i, placement) for i, layer in zip(ids, layers)]
    return blocks

def compute_loss(block, output, target, loss_fn):
    '''
    Computes the loss and correctly prepares the gradients for the pipelined backward pass
    '''
    output = output.detach()
    output.requires_grad = True
    loss = loss_fn(output, target, reduction="sum")
    loss.backward()
    block.grads.append((None, output.grad.data))
