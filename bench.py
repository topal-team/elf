import os
import torch
import torch.nn as nn
import torch.distributed as dist
from engine import StageScheduler
from schedule import generate_afab_schedule, generate_1f1b_schedule
from pipeline import pipeline_from_layers
from torchvision.models import resnet101
import time

import logging
logger = logging.getLogger(f'benchmark')
logging.basicConfig(level = logging.INFO) # 

if __name__ == '__main__':
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
    pipeline = pipeline_from_layers(my_layers, placement, global_rank)

    batch_size = 32
    n_iters = 50

    if global_rank == 0:
        start = time.time()
        logger.info(f'Starting benchmark for regular',)
        for _ in range(n_iters):
            dummy = torch.randn((batch_size, 3, 224, 224)).cuda()
            target = torch.randn((batch_size, 1000)).cuda()
            y = model(dummy)
            loss = torch.nn.functional.cross_entropy(y, target)
            loss.backward()
        logger.info(f'Regular : {time.time() - start:.3f}s')

    schedule = generate_afab_schedule(placement, batch_size)
    scheduler = StageScheduler(schedule, pipeline)

    start = time.time()
    if global_rank == 0: logger.info(f'Starting benchmark for AFAB')
    for _ in range(n_iters):
        dummy = torch.randn((batch_size, 3, 224, 224)).cuda()
        target = torch.randn((batch_size, 1000)).cuda()
        scheduler.train_step(dummy, target, torch.nn.functional.cross_entropy)
    if global_rank == 0: logger.info(f'AFAB : {time.time() - start:.3f}s')

    schedule = generate_1f1b_schedule(placement, batch_size)
    scheduler = StageScheduler(schedule, pipeline)

    start = time.time()
    if global_rank == 0: logger.info(f'Starting benchmark for 1F1B')
    for _ in range(n_iters):
        dummy = torch.randn((batch_size, 3, 224, 224)).cuda()
        target = torch.randn((batch_size, 1000)).cuda()
        scheduler.train_step(dummy, target, torch.nn.functional.cross_entropy)
    if global_rank == 0: logger.info(f'1F1B : {time.time() - start:.3f}s')

    if dist.is_initialized():
        dist.destroy_process_group()
    
