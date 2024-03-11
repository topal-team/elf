import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from collections import deque

DEBUG = "DEBUG" in os.environ and os.environ["DEBUG"] != "0"

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

        # We need to keep track of ids when we merge layers together
        self.previous_id = None
        self.next_id = None

        # TODO : make this generic to any shape. With metadata ?
        self.input_shape = (self.model.in_features,)
        self.output_shape = (self.model.out_features,)

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

        tag = self.next_id
        activations = self.act_to_send.popleft()

        if DEBUG: print(f'{self} - Sending activations to layer {self.next_id} (tagged {tag}) on rank {self.next}')
        dist.send(activations, self.next, tag=tag)

    def send_backward(self):
        '''
        Send one gradient to the previous layer in the model
        '''
        if self.previous is None or len(self.grads_to_send) == 0: return

        # trick : to differentiate backward and forward passes, we use bitwise not tag for backward
        # We need that in case a layer has the same next and previous ranks ; otherwise we cannot know if we received activations or gradients
        # Since all ids are positive, there is no collision
        tag = ~self.previous_id
        grads = self.grads_to_send.popleft()

        if DEBUG: print(f'{self} - Sending gradients to layer {self.previous_id} (tagged {tag}) on rank {self.previous}')
        dist.send(grads, self.previous, tag=tag)

    def recv_forward(self):
        '''
        Receive and store one activation to forward
        '''
        if self.previous is None: return

        tag = self.previous_id + 1

        buffer = torch.empty(self.input_shape).cuda()

        if DEBUG: print(f'{self} - Waiting for activations from layer {self.previous_id} (tagged {tag}) on rank {self.previous}')
        dist.recv(buffer, self.previous, tag=tag)
        if DEBUG: print(f'{self} - Received activations !')

        self.inputs.append(buffer.clone().detach())

    def recv_backward(self):
        '''
        Receive and store one gradient to backward
        '''
        if self.next is None: return

        tag = ~(self.next_id - 1) # see trick in send_backward

        buffer = torch.empty(self.output_shape).cuda()

        if DEBUG: print(f'{self} - Waiting for gradients from layer {self.next_id} (tagged {tag}) on rank {self.next}')
        dist.recv(buffer, self.next, tag=tag)
        if DEBUG: print(f'{self} - Received gradients !')

        self.grads.append(buffer)

    def merge(self, block):
        '''
        Merge another block with this one
        '''
        self.model = nn.Sequential(self.model, block.model)
        self.next = block.next
        self.next_id = block.next_id


if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    # Suppose this is our model partition
    placement = torch.randint(0, world_size, (4,)).cuda()
    # placement = torch.tensor([0, 1, 0, 3, 2, 1, 2, 3]).cuda() # test
    dist.broadcast(placement, 0) # synchronize placement on all processes
    print(f'Placement : {placement}')

    blocks = []
    all_layers = [] # for tests
    for i in range(len(placement)):
        # We share weights to all devices but only one will use it + the last one to reconstruct the entire model
        layer = nn.Linear(3, 3, bias=False).cuda()
        weights = layer.state_dict()['weight']
        dist.broadcast(weights, src = 0)

        if global_rank == placement[i]:
            layer.load_state_dict({"weight": weights})
            block = PipelineBlock(layer, global_rank, i)
            if i < len(placement) - 1:
                block.next = placement[i + 1]
                block.next_id = i + 1
            if i > 0:
                block.previous = placement[i - 1]
                block.previous_id = i - 1
            blocks.append(block)

        if global_rank == placement[-1]:
            layer.load_state_dict({"weight": weights})
            all_layers.append(layer)

    # Merge contiguous blocks that are on the same device
    i = 0
    while i < len(blocks) - 1:
        block = blocks[i]
        if block.rank == block.next:
            if DEBUG: print(f'Rank {global_rank} - Merging block {i} and {i + 1}')
            block.merge(blocks[i + 1])
            blocks.pop(i + 1)
            i -= 1
        i += 1

    batch = torch.randn((len(placement), 3)).cuda()
    dist.broadcast(batch, src = 0) # for tests
    target = torch.randn_like(batch).cuda() # No need to broadcast since only last device will use this anyway
    splits = iter(batch.split(1, dim=0))

    # Schedule here !
    # All forward
    result = []
    for b in blocks:
        for i in range(batch.size(0)):
            # Feed micro-batch in pipeline
            if b.previous is None:
                if DEBUG: print(f'{b} - Feeding micro batch {i} into the pipeline')
                b.inputs.append(next(splits))

            b.recv_forward()
            b.forward()
            # Last layer has the result
            if b.next is None: result.append(b.act_to_send.popleft().unsqueeze(0))
            b.send_forward()

    grads = []
    # All backward
    for b in reversed(blocks):
        for i in range(batch.size(0)):
            # Compute loss on last device
            if b.next is None:
                if DEBUG: print(f'{b} - Starting backward pass for micro batch {i}')
                output = result[i].detach()
                output.requires_grad = True
                loss = F.mse_loss(output, target[i].unsqueeze(0), reduction="sum")
                loss.backward()
                b.grads.append(output.grad.data)

            b.recv_backward()
            b.backward()
            if b.previous is None: grads.append(b.grads_to_send.popleft())
            b.send_backward()

    # dist.barrier()

    for i,b in enumerate(blocks):
        assert len(b.activations) == 0, f'{b} - Should be no activation left, {len(b.activations)} still in queue'
        assert len(b.inputs) == 0, f'{b} - Should be no input left to computet, {len(b.inputs)} still in queue'
        assert len(b.act_to_send) == 0, f'{b} - Should be no activation left to send, {len(b.act_to_send)} still in queue'
        assert len(b.grads) == 0, f'{b} - Should be no gradients left, {len(b.grads)} still in queue'
        assert len(b.inputs_to_keep) == 0, f'{b} - Should be no inputs left to backward, {len(b.inputs_to_keep)} still in queue'
        assert len(b.grads_to_send) == 0, f'{b} - Should be no grads left to send, {len(b.grads_to_send)} still in queue'

    if global_rank == placement[-1]: # last device has the result
        block = blocks[-1]
        output = torch.cat(result, dim=0)
        batch.requires_grad = True
        groundtruth = nn.Sequential(*all_layers).cuda()(batch)
        assert torch.allclose(output, groundtruth), f'Pipelined and regular models have different outputs : {output} and {groundtruth}'
        print(f'{block} - Outputs are correct :)')

        loss = F.mse_loss(groundtruth, target, reduction="sum")
        loss.backward()
        if placement[0] != placement[-1]: dist.send(batch.grad.data, placement[0]) # First layer has gradients of pipelined model, send to check

    if global_rank == placement[0]:
        block = blocks[0]
        grads = torch.cat(grads, dim=0)
        if placement[0] == placement[-1]: 
            groundtruth = batch.grad.data
        else:
            groundtruth = torch.empty_like(batch)
            dist.recv(groundtruth, placement[-1])
        assert torch.allclose(grads, groundtruth), f'Pipelined and regular models have different gradients : {grads} and {groundtruth}'
        print(f'{block} - Gradients are correct :))')

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
