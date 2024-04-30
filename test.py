import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
from pipeline.pipeline import Pipeline
from argparse import ArgumentParser

import logging
logger = logging.getLogger(f'main')
logging.basicConfig(level = logging.DEBUG)

def create_model():
    all_layers = [nn.Linear(3, 3, bias=False) for i in range(32)]

    model = nn.Sequential(*all_layers)
    torch.save(model, "test-model.pt")

def load_full_model(size, path="test-model.pt"):
    '''
    Loads the sequential model stored in `path`, and returns the `size` first layers.
    '''
    model = torch.load(path)
    new_model = nn.Sequential(*list(model.children())[:size])
    return nn.Sequential(new_model)

def load_parts_model(placement, global_rank, path="test-model.pt"):
    '''
    Loads the sequential model stored in `path`, and returns the layers corresponding to `global_rank` in the model placement.
    '''
    indices = [idx for idx, p in enumerate(placement) if global_rank == p]
    model = torch.load(path)
    children = list(model.children())
    blocks = [children[idx] for idx in indices]
    return blocks

def test_pipeline(blocks, placement, scheduler, batch_size):
    '''
    Test that a schedule computes the forward & backward passes correctly
    To do that, we compute from a random sample and compare with the results/gradients from the same model, fully reconstruced on a single device
    '''

    split_size = 1
    batch = torch.randn((batch_size, 3)).cuda()

    # Pipelined model will use the batch from its first rank
    # While full model will use the batch from the last rank of the pipelined model
    # So we have to sync them
    if placement[0] != placement[-1]:
        if global_rank == placement[0]: dist.send(batch, dst = placement[-1])
        if global_rank == placement[-1]: dist.recv(batch, src = placement[0])
        
    target = torch.randn_like(batch).cuda() # No need to share since only last device will use this anyway

    pipe = Pipeline(blocks, placement, partition = None, schedule = scheduler)
    output = pipe(batch, target, F.mse_loss, split_size)
    blocks = pipe.blocks # we shouldn't access directly the internal modules but it's for the purpose of testing
    grads = None
    if global_rank == placement[0]:
        grads = list(blocks[0].grads_to_send)
        blocks[0].grads_to_send.clear()
    logger.debug(f'[Rank {global_rank}] : result = {output}, grads = {grads}')
    
    for b in blocks:
        assert len(b.activations) == 0, f'{b} - There should be no activation left, {len(b.activations)} still in queue'
        assert len(b.inputs) == 0, f'{b} - There should be no input left to compute, {len(b.inputs)} still in queue'
        assert len(b.act_to_send) == 0, f'{b} - There should be no activation left to send, {len(b.act_to_send)} still in queue'
        assert len(b.grads) == 0, f'{b} - There should be no gradients left, {len(b.grads)} still in queue'
        assert len(b.inputs_to_keep) == 0, f'{b} - There should be no inputs left to backward, {len(b.inputs_to_keep)} still in queue'
        assert len(b.grads_to_send) == 0, f'{b} - There should be no grads left to send, {len(b.grads_to_send)} still in queue'

    # Last device has the result, we reconstruct the full model on this one for simplicity
    if global_rank == placement[-1]:
        block = blocks[-1]
        batch.requires_grad = True
        full_model = load_full_model(len(placement)).cuda()
        groundtruth = full_model(batch)
        assert torch.allclose(output, groundtruth, rtol=1e-3, atol=1e-6), f'Pipelined and regular models have different outputs : {output} and {groundtruth}'
        logger.info(f'{block} - Outputs are correct :)')

        loss = F.mse_loss(groundtruth, target, reduction="sum") # If we use default reduction (mean) we need to also apply it on pipeline micro batches, which is not trivial
        loss.backward()
        # First device has the gradients w.r.t the inputs, it will check that they are right
        if placement[0] != placement[-1]: dist.send(batch.grad.data, placement[0])

    if global_rank == placement[0]:
        block = blocks[0]
        grads = torch.cat(grads, dim=0)
        if placement[0] == placement[-1]:
            groundtruth = batch.grad.data
        else:
            groundtruth = torch.empty_like(batch)
            dist.recv(groundtruth, placement[-1])

        assert torch.allclose(grads, groundtruth, rtol=1e-3, atol=1e-6), f'Pipelined and regular models have different gradients : {grads} and {groundtruth}'
        logger.info(f'{block} - Gradients are correct :))')

if __name__ == "__main__":
    parser = ArgumentParser(description = "Demo/Test of pipelined model")
    parser.add_argument('--log', choices=['debug', 'info', 'none'], default='info', required=False, help="logging level")
    args = parser.parse_args()
    match args.log:
        case 'debug':
            logging.getLogger().setLevel(logging.DEBUG)
        case 'info':
            logging.getLogger().setLevel(logging.INFO)
        case 'none':
            logging.getLogger().setLevel(100)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    # Suppose this is our model partition : a model is a sequence of submodules [0, 1, ..., n], and each submodule i is placed on (global) rank placement[i]
    # placement = torch.randint(0, world_size, (4,)).cuda()

    # Load your model here (each process should load the right layers depending on placement)

    placements = [
        [0, 1, 2, 3],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        [0, 1, 1, 0], # Hanayo style
        [0, 1, 2, 3, 3, 2, 1, 0],
        [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]
    ]

    batch_sizes = [4, 8, 16, 32]

    # Check that the results and gradients are the same as a single-gpu model
    for p in placements:
        for b in batch_sizes:
            if b < len(p): continue
            if global_rank == 0: logger.info(f'Testing placement {p} with batch size {b}')
            layers = load_parts_model(p, global_rank)
            test_pipeline(layers, p, "afab", b)
            dist.barrier()
            if global_rank == 0: logger.info('\n')
            
    if dist.is_initialized():
        dist.destroy_process_group()
