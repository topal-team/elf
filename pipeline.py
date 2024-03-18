import torch
import torch.nn as nn
import torch.distributed as dist
import os
from collections import deque

DEBUG = "DEBUG" in os.environ and os.environ["DEBUG"] != "0"

class TensorMetadata():
    MAX_SIZE = 64

    @staticmethod
    def from_tensor(t):
        shape = []
        assert len(t.shape) == 1, "Metadata should only have one dimension"
        for s in t:
            shape.append(s)
        return TensorMetadata(s)

    def __init__(self, t):
        self.shape = t.shape

    def to_tensor(self):
        t = torch.empty(TensorMetadata.MAX_SIZE).cuda()
        for i, s in enumerate(self.shape):
            t[i] = s
        return t
    
    def get_buffer(self):
        buffer = torch.empty(self.shape).cuda()
        return buffer

class PipelineBlock():
    '''
    Manages one layer/group of contiguous layers placed on one device
    '''
    def __init__(self, model, rank, id_):
        super(PipelineBlock, self).__init__()
        # Block infos
        self.model = model.cuda()
        self.rank = rank # global rank
        self.id = id_ # rank in the model

        # Queues of tensor to process
        self.inputs = deque() # Waiting for forward
        self.activations = deque() # Kept for backward
        self.grads = deque() # Waiting for backward
        self.act_to_send = deque() # Sent to next block
        self.grads_to_send = deque() # Sent to previous block
        self.inputs_to_keep = deque() # Kept for backward

        # Ranks where the previous/next blocks in the model are placed
        self.previous = None
        self.next = None

    def __str__(self) -> str:
        return f'[Layer {self.id} : GPU {self.rank}]'

    def forward(self):
        '''
        Perform the forward pass for one tensor of activations and register it as computed
        '''
        if len(self.inputs) == 0: return
        if DEBUG: print(f'{self} - Computing forward')
        x = self.inputs.popleft()
        x.requires_grad = True
        y = self.model(x)
        self.activations.append(y)
        self.act_to_send.append(y)
        self.inputs_to_keep.append(x)
        
    def backward(self):
        '''
        Perform the backward pass for one tensor of gradients and register it as computed
        Backward assumes activations AND grads to be on top of the queue
        '''
        if len(self.grads) == 0: return
        if DEBUG: print(f'{self} - Computing backward')

        act = self.activations.popleft()
        grads = self.grads.popleft()
        x = self.inputs_to_keep.popleft()

        (act * grads).sum().backward()

        self.grads_to_send.append(x.grad.data)

    def send_forward(self):
        '''
        Send one activation to the next layer in the model
        '''
        if self.next is None or len(self.act_to_send) == 0: return

        tag = self.id + 1
        activations = self.act_to_send.popleft()

        metadata = TensorMetadata(activations).to_tensor()
        dist.send(metadata, self.next, tag=tag)

        if DEBUG: print(f'{self} - Sending activations to layer {self.id + 1} (tagged {tag}) on rank {self.next}')
        dist.send(activations, self.next, tag=tag)

    def send_backward(self):
        '''
        Send one gradient to the previous layer in the model
        '''
        if self.previous is None or len(self.grads_to_send) == 0: return

        # trick : to differentiate backward and forward passes, we use bitwise not tag for backward
        # We need that in case a layer has the same next and previous ranks ; otherwise we cannot know if we received activations or gradients
        # Since all ids are positive, there is no collision
        tag = ~(self.id - 1)
        grads = self.grads_to_send.popleft()

        metadata = TensorMetadata(grads).to_tensor()
        dist.send(metadata, self.previous, tag=tag)

        if DEBUG: print(f'{self} - Sending gradients to layer {self.id - 1} (tagged {tag}) on rank {self.previous}')
        dist.send(grads, self.previous, tag=tag)

    def recv_forward(self):
        '''
        Receive and store one activation to forward
        '''
        if self.previous is None: return

        tag = self.id

        buffer = torch.empty(TensorMetadata.MAX_SIZE).cuda()
        dist.recv(buffer, self.previous, tag=tag)
        metadata = TensorMetadata.from_tensor(buffer)
        buffer = metadata.get_buffer()

        if DEBUG: print(f'{self} - Waiting for activations from layer {self.id - 1} (tagged {tag}) on rank {self.previous}')
        dist.recv(buffer, self.previous, tag=tag)
        if DEBUG: print(f'{self} - Received activations !')

        self.inputs.append(buffer.clone().detach())

    def recv_backward(self):
        '''
        Receive and store one gradient to backward
        '''
        if self.next is None: return

        tag = ~self.id # see trick in send_backward

        buffer = torch.empty(TensorMetadata.MAX_SIZE).cuda()
        dist.recv(buffer, self.next, tag=tag)
        metadata = TensorMetadata.from_tensor(buffer)
        buffer = metadata.get_buffer()

        if DEBUG: print(f'{self} - Waiting for gradients from layer {self.id + 1} (tagged {tag}) on rank {self.next}')
        dist.recv(buffer, self.next, tag=tag)
        if DEBUG: print(f'{self} - Received gradients !')

        self.grads.append(buffer)

    def merge(self, block):
        '''
        Merge another block with this one
        '''
        self.model = nn.Sequential(self.model, block.model)
        self.next = block.next

def pipeline_from_layers(layers, placement, global_rank):
    ids = [idx for idx, p in enumerate(placement) if global_rank == p]
    blocks = []
    for id_, layer in zip(ids, layers):
        block = PipelineBlock(layer.cuda(), global_rank, id_)
        if id_ < len(placement) - 1:
            block.next = placement[id_ + 1]
        if id_ > 0:
            block.previous = placement[id_ - 1]
        blocks.append(block)

    # Merge contiguous blocks that are on the same device
    i = 0
    while i < len(blocks) - 1:
        block = blocks[i]
        if block.rank == block.next:
            if DEBUG: print(f'Rank {global_rank} - Merging block {i} and {i + 1}')
            block.merge(blocks[i + 1])
            blocks.pop(i + 1)
            for j in range(i, len(blocks)):
                blocks[j].id -= 1
            i -= 1
        i += 1

    return blocks

def compute_loss(block, output, target, loss_fn):
    output = output.detach()
    output.requires_grad = True
    loss = loss_fn(output, target.unsqueeze(0), reduction="sum")
    loss.backward()
    block.grads.append(output.grad.data)
