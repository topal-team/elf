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

    split_sizes = [1, 2, 4, 8, 16, 32, 64]
    times = []
    for size in split_sizes:
        if global_rank == 0: logger.info(f'Beginning bench for micro batches of size {size}')

        pipe = Pipeline(model, placement, schedule = "1f1b")

        # Warmup
        for i in range(5):
            if global_rank == 0: logger.info(f'Warmup {i}')
            _ = pipe(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), size)
        start = time.time()
        for i in range(iters):
            if global_rank == 0: logger.info(f'Iter {i}')
            _ = pipe(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), size)
        end = time.time()
        t = (end - start) / iters
        if global_rank == 0:
            print(f'Time taken by custom pipe (size {size}) : {end - start:.2f}s. Average : {t:.3f}s')
            f.write(f'{size},{t}\n')
        times.append(t)

    if global_rank == 0: f.close()
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
