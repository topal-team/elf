import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
import time
from models.GPT import GPT
from pipeline.pipeline import create_pipeline, partition_model
from pipeline.schedule import generate_afab_schedule, generate_1f1b_schedule
from pipeline.engine import StageScheduler
from argparse import ArgumentParser
from settings import *
import psutil

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
    schedule = generate_1f1b_schedule(placement, block_size)
    stage = StageScheduler(schedule, blocks)

    # Warmup
    for _ in range(5):
        _ = stage.train_step(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum())
    start = time.time()
    for _ in range(iters):
        _ = stage.train_step(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum())
    end = time.time()
    print(f'Time taken by custom pipeline : {end - start:.2f}s. Average : {(end - start) / iters:.3f}s')
    # Time taken by custom pipeline : 16.79s. Average : 0.336s (AFAB XXXL)
    # Time taken by custom pipeline : 15.60s. Average : 0.312s (1F1B XXXL)

    # Time taken by custom pipeline : 3.61s. Average : 0.072s (AFAB ResNet Bottleneck 8)
    # Time taken by custom pipeline : 3.48s. Average : 0.070s 1F1B ResNet Bottleneck 8)

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


