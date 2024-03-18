import torch
import torch.nn.functional as F
import torch.distributed as dist
import os
from pipeline import pipeline_from_layers
from schedule import train_step_afab, train_step_1f1b
from test_model import load_full_model, load_parts_model

DEBUG = "DEBUG" in os.environ and os.environ["DEBUG"] != "0"

def test_pipeline(blocks, placement, schedule=train_step_afab):
    batch = torch.randn((len(placement), 3)).cuda()
    if global_rank == 0: dist.send(batch, dst = placement[-1]) # for tests
    if global_rank == placement[-1]: dist.recv(batch, src = 0)
    target = torch.randn_like(batch).cuda() # No need to share since only last device will use this anyway

    result, grads = schedule(blocks, batch, target, F.mse_loss)
    
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
        full_model = load_full_model(len(placement)).cuda()
        groundtruth = full_model(batch)
        assert torch.allclose(output, groundtruth), f'Pipelined and regular models have different outputs : {output} and {groundtruth}'
        print(f'{block} - Outputs are correct :)')

        loss = F.mse_loss(groundtruth, target, reduction="sum") # If we use default reduction (mean) we need to also apply it on pipeline micro batches, which is not trivial
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
        assert torch.allclose(grads, groundtruth, rtol=1e-3, atol=1e-6), f'Pipelined and regular models have different gradients : {grads} and {groundtruth} biggest difference is {torch.max((grads - groundtruth).abs())}'
        print(f'{block} - Gradients are correct :))')

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    # Suppose this is our model partition : a model is a sequence of submodules [0, 1, ..., n], and each submodule i is placed on rank placement[i]
    placement = torch.randint(0, world_size, (4,)).cuda()
    dist.broadcast(placement, 0) # synchronize placement on all processes
    print(f'Placement : {placement}')

    # Load your model here (each process should load the right layers depending on placement)
    layers = load_parts_model(placement, global_rank)
    # This method automatically creates the pipeline based on your placement. `blocks` now contains the pipelined blocks.
    blocks = pipeline_from_layers(layers, placement, global_rank)
    # After that you can use the schedules in `schedule.py` to compute results + gradients of your layers

    # Check that the results and gradients are the same as a single-gpu model
    test_pipeline(blocks, placement, schedule=train_step_afab)

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
