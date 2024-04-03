import os
import torch
import torch.nn as nn
import torch.distributed as dist
from engine import StageScheduler
from schedule import generate_afab_schedule, generate_1f1b_schedule
from pipeline import create_pipeline
from torchvision.models import resnet101
import time

from argparse import ArgumentParser
import logging
logger = logging.getLogger(f'benchmark')
logging.basicConfig(level = logging.INFO)

if __name__ == '__main__':
    parser = ArgumentParser(description = "Benchmark of pipeline schedules")
    parser.add_argument('--log', choices=['debug', 'info', 'none'], default='info', required=False, help="logging level")
    args = parser.parse_args()
    match args.log:
        case 'debug':
            logger.setLevel(logging.DEBUG)
        case 'info':
            logger.setLevel(logging.INFO)
        case 'none':
            logger.setLevel(100)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    model = resnet101().cuda()

    if global_rank == 0:
        my_layers = [nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)]

    elif global_rank == 1:

        my_layers = [nn.Sequential(model.layer2)]
    elif global_rank == 2:

        my_layers = [nn.Sequential(model.layer3)]
    else:

        my_layers = [nn.Sequential(model.layer4, model.avgpool, nn.Flatten(), model.fc)]
    
    placement = list(range(world_size))

    pipeline = create_pipeline(my_layers, placement)

    batch_size = 4
    split_size = 2
    n_iters = 1

    if global_rank == 0:
        start = time.time()
        logger.info(f'Starting benchmark for regular')
        for _ in range(n_iters):
            dummy = torch.randn((batch_size, 3, 224, 224)).cuda()
            target = torch.randn((batch_size, 1000)).cuda()
            y = model(dummy)
            loss = torch.nn.functional.cross_entropy(y, target)
            loss.backward()
        logger.info(f'Regular : {time.time() - start:.3f}s')
    
    schedule = generate_afab_schedule(placement, batch_size // split_size)
    scheduler = StageScheduler(schedule, pipeline)
    
    start = time.time()
    if global_rank == 0: logger.info(f'Starting benchmark for AFAB')
    for _ in range(n_iters):
        dummy = torch.randn((batch_size, 3, 224, 224)).cuda()
        target = torch.randn((batch_size, 1000)).cuda()
        scheduler.train_step(dummy, target, torch.nn.functional.cross_entropy, split_size)
    if global_rank == 0: logger.info(f'AFAB : {time.time() - start:.3f}s')
    
    schedule = generate_1f1b_schedule(placement, batch_size // split_size)
    scheduler = StageScheduler(schedule, pipeline)

    start = time.time()
    if global_rank == 0: logger.info(f'Starting benchmark for 1F1B')
    for _ in range(n_iters):
        dummy = torch.randn((batch_size, 3, 224, 224)).cuda()
        target = torch.randn((batch_size, 1000)).cuda()
        scheduler.train_step(dummy, target, torch.nn.functional.cross_entropy, split_size)
    if global_rank == 0: logger.info(f'1F1B : {time.time() - start:.3f}s')

    if dist.is_initialized():
        dist.destroy_process_group()
    
