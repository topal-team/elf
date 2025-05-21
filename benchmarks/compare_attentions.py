
import os
import sys
sys.path.append('../')

from models.simple import FullTransformer, ChainTransformer
from elf.utils import TimerGPU
from elf import Pipeline
from elf.pipeline import get_sources_targets_sequential


import torch
import torch.distributed as dist
import datetime
import torch.cuda.profiler as profiler
from torch.cuda import cudart
import torch.nn as nn

import argparse

def get_blocks(rank, placements, model):
    return [b for (i,b) in zip(placements, model.blocks) if i==rank]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--sdp_backend', type=str, default=None,
                        choices=[None, 'FLASH_ATTENTION', 'MATH'])
    parser.add_argument('--transformer_type', type=str, default='chain',
                        choices=['full', 'chain'])
    return parser.parse_args()

if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ws = int(os.getenv("WORLD_SIZE"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=240))

    args = parse_args()

    nmb = ws * 2
    mb_size = 8
    batch_size = nmb * mb_size

    num_heads = 16
    seq_len = 1024
    head_dim = 64
    n_blocks = 4
    embed_dim = num_heads * head_dim
    hidden_dim = embed_dim
    placements = [0, 1, 2, 3, 4, 5, 6, 7][:4]
    print(args.sdp_backend)
    
    match args.dtype:
        case 'float32': dtype = torch.float32
        case 'float16': dtype = torch.float16
        case 'bfloat16': dtype = torch.bfloat16

    if torch.cuda.is_available():
        # device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        device = local_rank
    else:
        device = torch.device("cpu")

    if args.transformer_type == 'full':
        model = (
            FullTransformer(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                n_blocks=n_blocks,
                seq_len=seq_len,
                num_heads=num_heads,
                sdp_backend=args.sdp_backend,
            )
            .to(dtype))
    else:
        model = (
		ChainTransformer(
			hidden_dim=hidden_dim,
			n_blocks=n_blocks,
			seq_len=seq_len,
			num_heads=num_heads,
			sdp_backend=args.sdp_backend,
		)
		.to(dtype))


    sample = model.get_sample(batch_size).to(device)  # keep as long for embedding
    target = model.get_target(batch_size).to(device)

    if args.transformer_type=='chain':
        sample = sample.to(dtype)
        target = target.to(dtype)

    rank_blocks = get_blocks(rank, placements, model)
    sources, targets = get_sources_targets_sequential(placements)

    pipe = Pipeline(rank_blocks, sample, schedule="1f1b", partitioner=False, placement=placements, sources=sources, targets=targets)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(pipe.parameters())

    print('WARM WARM UP')
    # Warmup iterations
    for _ in range(3):
        _ = pipe(sample, target, loss_fn, split_size=mb_size, profile=False)
        optimizer.step()

    torch.cuda.synchronize()
    
    # Profile iterations
    profiler.start()
    cudart().cudaProfilerStart()

    with TimerGPU() as timer:
        for i in range(30):
            if rank == 0:
                print(f"Iteration {i}")
            _ = pipe(sample, target, loss_fn, split_size=mb_size, profile=False)
            optimizer.step()

    print(f'Rank: {rank}, time: {timer.time()}')
    cudart().cudaProfilerStop()
    profiler.stop()

    pipe.clear()

    if dist.is_initialized():
        # dist.barrier()
        dist.destroy_process_group()