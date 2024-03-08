import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from collections import deque

class DeviceState():
    def __init__(self, blocks) -> None:
        # We assume that blocks are sorted by order in the model
        self.blocks = blocks

    def send_forward(self, block):
        '''
        Send activation for one micro batch computed by block to its next layer
        '''
        if block.next is None: return
        idx = block.current_fwd - len(block.act_to_send)
        activation = block.act_to_send.popleft()
        print(f'{block} - Sending activations to rank {block.next}')
        dist.send(activation, block.next, tag=idx)

    def send_backward(self, block):
        '''
        Send gradients for one micro batch computed by block to its previous layer
        '''
        if block.previous is None: return
        idx = block.current_bwd - len(block.grads_to_send)
        grads = block.grads_to_send.popleft()
        print(f'{block} - Sending gradients to rank {block.previous}')
        dist.send(grads, block.previous, tag=idx)

    def recv_forward(self):
        '''
        Receives a forward tensor and places it in the right queue.
        After the call to this function, the device will have something to compute for one of its layers (unless everything is finished already) but we cannot know in advance the for which layer it will be.
        '''
        tag = -1
        for block in self.blocks:
            if block.previous is None: continue
            buffer = torch.empty(block.input_shape).cuda()
            print(f'{block} - Waiting for activations from rank {block.previous}')
            dist.recv(buffer, src=block.previous, tag=tag) # UNSAFE as we're not sure about the shape in advance
            for b in self.blocks:
                # Is this always true ?
                if b.previous == block.previous and b.current_fwd == tag:
                    print(f'{b} - Received new activations !')
                    b.inputs.append(buffer)
                    b.current_fwd += 1
                    return

    def recv_backward(self):
        '''
        Receives a backward tensor and places it in the right queue.
        After the call to this function, the device will have something to compute for one of its layers (unless everything is finished already) but we cannot know in advance the for which layer it will be.
        '''
        tag = -1
        for block in self.blocks:
            if block.next is None: continue
            buffer = torch.empty(block.output_shape).cuda()
            print(f'{block} - Waiting for gradients from rank {block.next}')
            dist.recv(buffer, src=block.next, tag=tag) # UNSAFE as we're not sure about the shape in advance
            for b in self.blocks:
                # Is this always true ?
                if b.next == block.next and b.current_bwd == tag:
                    print(f'{b} - Received new gradients !')
                    b.grads.append(buffer)
                    b.current_bwd += 1
                    return

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
        self.inputs = deque()
        self.activations = deque()
        self.grads = deque()
        self.act_to_send = deque()
        self.grads_to_send = deque()

        # Ranks where the previous/next blocks in the model are placed
        self.previous = None 
        self.next = None

        # Keep track of the micro batches
        self.current_fwd = 0 # idx of next micro batch to forward
        self.current_bwd = 0

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
        x = self.inputs[-1]
        x.requires_grad = True
        assert self.inputs[-1].requires_grad, "Why don't you need gradients ??"
        y = self.model(x)
        self.activations.append(y)
        self.act_to_send.append(y)

    def backward(self):
        '''
        Perform the backward pass for one tensor of gradients and register it as computed
        Backward assumes activations AND grads to be on top of the queue
        '''
        if len(self.grads) == 0: return
        print(f'{self} - Computing backward')

        act = self.activations.popleft()
        grads = self.grads.popleft()
        (act * grads).sum().backward()

        x = self.inputs.popleft()
        self.grads_to_send.append(x.grad.data)

    def send_forward(self):
        if self.next is None or len(self.act_to_send) == 0: return

        tag = self.id + 1
        activations = self.act_to_send.popleft()

        print(f'{self} - Sending activations to layer {tag} (tagged) on rank {self.next}')
        dist.send(activations, self.next, tag=tag)

    def send_backward(self):
        if self.previous is None or len(self.grads_to_send) == 0: return

        tag = self.id - 1
        grads = self.grads_to_send.popleft()

        print(f'{self} - Sending gradients to layer {tag} (tagged) on rank {self.previous}')
        dist.send(grads, self.previous, tag=tag)

    def recv_forward(self):
        if self.previous is None: return

        tag = self.id - 1
        buffer = torch.empty(self.input_shape).cuda()

        print(f'{self} - Waiting for activations from layer {tag} (tagged) on rank {self.previous}')
        dist.recv(buffer, self.previous, tag=tag)
        print(f'{self} - Received activations !')
        # buffer.requires_grad = True
        self.inputs.append(buffer)

    def recv_backward(self):
        if self.next is None: return

        tag = self.id + 1
        buffer = torch.empty(self.output_shape).cuda()

        print(f'{self} - Waiting for gradients from layer {tag} (tagged) on rank {self.next}')
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
    # placement = torch.randint(0, world_size, (4,)).cuda()
    placement = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).cuda() # test
    dist.broadcast(placement, 0) # synchronize placement on all processes
    print(f'Placement : {placement}')

    blocks = []
    for i in range(len(placement)):
        if placement[i] == global_rank:
            block = PipelineBlock(nn.Linear(3, 3).cuda(), global_rank, i)
            block.next = placement[i + 1] if i < len(placement) - 1 else None
            block.previous = placement[i - 1] if i > 0 else None
            assert block.next != block.previous, "Cannot receive from the same device for forward and backward (not supported yet)"
            blocks.append(block)

    state = DeviceState(blocks)

    batch = torch.randn((16, 3))
    splits = iter(batch.split(1, dim=0))

    # Schedule here !
    # All forward
    for b in blocks:
        for i in range(batch.size(0)):
            # Feed micro-batch in pipeline
            if b.id == 0:
                print(f'{b} - Feeding micro batch {i} into the pipeline')
                micro_batch = next(splits).cuda()
                # micro_batch.requires_grad = True
                b.inputs.append(micro_batch)
                b.current_fwd += 1
            b.recv_forward()
            b.forward()
            b.send_forward()

    # All backward
    for b in reversed(blocks):
        for i in range(batch.size(0)):
            # Compute loss on last device
            if b.id == len(placement) - 1:
                print(f'{b} - Starting backward pass for micro batch {i}')
                output = b.activations[-1].detach()
                output.requires_grad = True
                loss = F.mse_loss(output, torch.randn(3).cuda())
                loss.backward()
                b.grads.append(output.grad.data)
                b.current_bwd += 1

            b.recv_backward()
            b.backward()
            b.send_backward()

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
