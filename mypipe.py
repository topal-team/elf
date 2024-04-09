import torch
import torch.distributed as dist
import os
import time
from pipeline.pipeline import create_pipeline, partition_model
from pipeline.schedule import generate_afab_schedule, generate_1f1b_schedule
from pipeline.engine import StageScheduler
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

    sequence = partition_model(model, placement = placement)
    blocks = create_pipeline(sequence, placement)

    nmb = [1, 2, 4, 8, 16, 32, 64]
    times = []
    for n_micro_batches in nmb:
        if global_rank == 0: logger.info(f'Beginning bench for {n_micro_batches} micro batches (split size = {batch_size // n_micro_batches})')
        schedule = generate_afab_schedule(placement, n_micro_batches)
        stage = StageScheduler(schedule, blocks)

        # Warmup
        for i in range(5):
            if global_rank == 0: logger.info(f'Warmup {i}')
            _ = stage.train_step(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), batch_size // n_micro_batches)
        start = time.time()
        for i in range(iters):
            if global_rank == 0: logger.info(f'Iter {i}')
            _ = stage.train_step(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), batch_size // n_micro_batches)
        end = time.time()
        t = (end - start) / iters
        if global_rank == 0:
            print(f'Time taken by custom pipe ({n_micro_batches} micro batches) : {end - start:.2f}s. Average : {t:.3f}s')
            f.write(f'{batch_size // n_micro_batches},{t}')
        times.append(t)

    if global_rank == 0: f.close()
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
