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

    inputs = inputs.cuda()

    sequence = partition_model(model, placement = placement)
    blocks = create_pipeline([sequence], placement)

    sizes = [1, 2, 4, 8, 16, 32]
    times = []
    for block_size in sizes:
        schedule = generate_1f1b_schedule(placement, dataset_size // block_size)
        stage = StageScheduler(schedule, blocks)

        # Warmup
        for _ in range(5):
            _ = stage.train_step(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), block_size)
        start = time.time()
        for _ in range(iters):
            _ = stage.train_step(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), block_size)
        end = time.time()
        t = (end - start) / iters
        print(f'Time taken by custom pipe (block size = {block_size}) : {end - start:.2f}s. Average : {t:.3f}s')
        times.append(t)

        if global_rank == 0:
            with open(f'custom_GPTXXXL_{dataset_size}.txt', 'w') as file:
                for size, t in zip(sizes, times):
                    file.write(f'{size}, {t}\n')
    
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


