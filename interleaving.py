import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from collections import deque

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

        # TODO : make this generic to other layers
        self.input_shape = (self.model.in_features,)
        self.output_shape = (self.model.out_features,)

    def __str__(self) -> str:
        return f'[Layer {self.id} : GPU {self.rank}]'

    def forward(self):
        '''
        Perform the forward pass for one tensor of activations and register it as computed
        '''
        if len(self.inputs) == 0: return
        print(f'{self} - Computing forward')
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
        print(f'{self} - Computing backward')

        act = self.activations.popleft()
        grads = self.grads.popleft()
        x = self.inputs_to_keep.popleft()

        (act * grads).sum().backward()

        self.grads_to_send.append(x.grad.data)

    def send_forward(self):
        if self.next is None or len(self.act_to_send) == 0: return

        tag = self.id + 1
        activations = self.act_to_send.popleft()

        print(f'{self} - Sending activations to layer {self.id + 1} (tagged {tag}) on rank {self.next}')
        dist.send(activations, self.next, tag=tag)

    def send_backward(self):
        if self.previous is None or len(self.grads_to_send) == 0: return

        # trick : to differentiate backward and forward passes, we use bitwise not tag for backward
        # We need that in case a layer has the same next and previous ; otherwise we cannot know if we received activations or gradients
        # Since all ids are positive, there is no collision
        tag = ~(self.id - 1)
        grads = self.grads_to_send.popleft()

        print(f'{self} - Sending gradients to layer {self.id - 1} (tagged {tag}) on rank {self.previous}')
        dist.send(grads, self.previous, tag=tag)

    def recv_forward(self):
        if self.previous is None: return

        tag = self.id
        buffer = torch.empty(self.input_shape).cuda()

        print(f'{self} - Waiting for activations from layer {self.id - 1} (tagged {tag}) on rank {self.previous}')
        dist.recv(buffer, self.previous, tag=tag)
        print(f'{self} - Received activations !')

        self.inputs.append(buffer.clone().detach())

    def recv_backward(self):
        if self.next is None: return

        tag = ~self.id # see trick in send_backward
        buffer = torch.empty(self.output_shape).cuda()

        print(f'{self} - Waiting for gradients from layer {self.id + 1} (tagged {tag}) on rank {self.next}')
        dist.recv(buffer, self.next, tag=tag)
        print(f'{self} - Received gradients !')

        self.grads.append(buffer)

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    # Suppose this is our model partition
    # placement = torch.randint(0, world_size, (8,)).cuda()
    placement = torch.tensor([0, 1, 0, 3, 2, 1, 2, 3]).cuda() # test
    dist.broadcast(placement, 0) # synchronize placement on all processes
    print(f'Placement : {placement}')

    blocks = []
    all_layers = [] # for tests
    for i in range(len(placement)):
        layer = nn.Linear(i+1, i+2, bias=False).cuda()
        weights = layer.state_dict()['weight']
        dist.broadcast(weights, src = 0)
        if global_rank == placement[i]:
            layer.load_state_dict({"weight": weights})
            block = PipelineBlock(layer, global_rank, i)
            block.next = placement[i + 1] if i < len(placement) - 1 else None
            block.previous = placement[i - 1] if i > 0 else None
            # assert block.next != block.previous, "Cannot receive from the same device for forward and backward (not supported yet)"
            blocks.append(block)

        if global_rank == placement[-1]:
            layer.load_state_dict({"weight": weights})
            all_layers.append(layer)

    batch = torch.randn((len(placement), 1)).cuda()
    dist.broadcast(batch, src = 0) # for tests
    target = torch.randn((len(placement), len(placement) + 1)).cuda() # No need to broadcast since only last device will use this anyway
    splits = iter(batch.split(1, dim=0))

    # Schedule here !
    # All forward
    result = []
    for b in blocks:
        for i in range(batch.size(0)):
            # Feed micro-batch in pipeline
            if b.id == 0:
                print(f'{b} - Feeding micro batch {i} into the pipeline')
                b.inputs.append(next(splits))

            b.recv_forward()
            b.forward()
            if b.id == len(placement) - 1: result.append(b.act_to_send.popleft().unsqueeze(0))
            b.send_forward()

    grads = []
    # All backward
    for b in reversed(blocks):
        for i in range(batch.size(0)):
            # Compute loss on last device
            if b.id == len(placement) - 1:
                print(f'{b} - Starting backward pass for micro batch {i}')
                output = result[i].detach()
                output.requires_grad = True
                loss = F.mse_loss(output, target[i].unsqueeze(0), reduction='sum')
                loss.backward()
                b.grads.append(output.grad.data)

            b.recv_backward()
            b.backward()
            if b.id == 0: grads.append(b.grads_to_send.popleft())
            b.send_backward()

    dist.barrier()
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
        dist.send(batch.grad.data, placement[0]) # First layer has gradients, send to check

    if global_rank == placement[0]:
        block = blocks[0]
        grads = torch.cat(grads, dim=0)
        groundtruth = torch.empty_like(batch)
        dist.recv(groundtruth, placement[-1])
        assert torch.allclose(grads, groundtruth), f'Pipelined and regular models have different gradients : {grads} and {groundtruth}'
        print(f'{block} - Gradients are correct :))')

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
