import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from collections import deque
from pipeline import PipelineBlock
from schedule import train_step_afab

DEBUG = "DEBUG" in os.environ and os.environ["DEBUG"] != "0"

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    # Suppose this is our model partition
    
    placement = torch.randint(0, world_size, (world_size*2,)).cuda()
    dist.broadcast(placement, 0) # synchronize placement on all processes
    print(f'Placement : {placement}')

    blocks = []
    all_layers = [] # for tests
    for i in range(len(placement)):
        # We share weights to all devices but only one will use it + the last one to reconstruct the entire model
        layer = nn.Linear(i + 1, i + 2, bias=False).cuda()
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
            for j in range(i, len(blocks)):
                blocks[j].id -= 1
            i -= 1
        i += 1

    batch = torch.randn((len(placement), 1)).cuda()
    dist.broadcast(batch, src = 0) # for tests
    target = torch.randn((len(placement), len(placement) + 1)).cuda() # No need to broadcast since only last device will use this anyway

    result, grads = train_step_afab(blocks, batch, target, F.mse_loss)
    
    for i,b in enumerate(blocks):
        assert len(b.activations) == 0, f'{b} - There should be no activation left, {len(b.activations)} still in queue'
        assert len(b.inputs) == 0, f'{b} - There should be no input left to compute, {len(b.inputs)} still in queue'
        assert len(b.act_to_send) == 0, f'{b} - There should be no activation left to send, {len(b.act_to_send)} still in queue'
        assert len(b.grads) == 0, f'{b} - There should be no gradients left, {len(b.grads)} still in queue'
        assert len(b.inputs_to_keep) == 0, f'{b} - There should be no inputs left to backward, {len(b.inputs_to_keep)} still in queue'
        assert len(b.grads_to_send) == 0, f'{b} - There should be no grads left to send, {len(b.grads_to_send)} still in queue'
    

    if global_rank == placement[-1]: # last device has the result
        block = blocks[-1]
        output = torch.cat(result, dim=0)
        batch.requires_grad = True
        groundtruth = nn.Sequential(*all_layers).cuda()(batch)
        assert torch.allclose(output, groundtruth, rtol=1e-4, atol=1e-7), f'Pipelined and regular models have different outputs : {output} and {groundtruth}'
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
        assert torch.allclose(grads, groundtruth, rtol=1e-4, atol=1e-7), f'Pipelined and regular models have different gradients : {grads} and {groundtruth}'
        print(f'{block} - Gradients are correct :))') # Sometimes there is a slight numerical instability

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
