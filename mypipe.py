import torch
import torch.distributed as dist
import os
import time
from pipeline.pipeline import Pipeline
from argparse import ArgumentParser
from settings import *

import logging
logger = logging.getLogger('gpt')
logging.basicConfig(level = logging.INFO)

if __name__ == "__main__":
    parser = ArgumentParser(description = "Pipelined model with custom engine")
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

    if global_rank == 0: f = open("mypipe.out", "w")

    inputs = inputs.cuda()

    times = []
    for size in split_sizes:
        if (batch_size // size) % len(placement) != 0:
            if global_rank == 0: logger.warning(f'The number of micro batches should be a multiple of the number of stages ! Got {batch_size // size} and {len(placement)}. Skipping.')
            continue
        if global_rank == 0: logger.info(f'Beginning bench for micro batches of size {size}')

        pipe = Pipeline(model, placement, schedule = schedule)
        
        # Warmup
        for i in range(5):
            if global_rank == 0: logger.info(f'Warmup {i}')
            _ = pipe(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), size)
        torch.cuda.reset_peak_memory_stats()
        
        iter_times = []
        for i in range(iters):
            if global_rank == 0: logger.info(f'Iter {i}')
            start = time.time()
            _ = pipe(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), size)
            end = time.time()
            iter_times.append(end - start)
        t = sorted(iter_times)[iters // 2] # median
        if global_rank == 0:
            print(f'Time taken by custom pipe (size {size}) : {end - start:.2f}s. Median : {t:.3f}s')
            f.write(f'{size},{t},{torch.cuda.max_memory_allocated() / (2**30)}\n')
        times.append(t)

    if global_rank == 0: f.close()
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
