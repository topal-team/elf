import os
import torch
import torch.distributed as dist
import sys
sys.path.append("./")
import pipeline.pipeline as MyPipe
from pipeline.utils import Timer
from settings import *

import torch.distributed.pipelining as PiPPy
from pippy import split_by_graph

placement = [0, 1, 2, 3]
schedule = "afab"

def pippy(model):
    policy = split_by_graph(len(placement))
    pipe = PiPPy.pipeline(
        model,
        len(placement),
        example_args=(inputs.clone(),),
        example_kwargs={},
        split_spec=None,
        split_policy=policy
    )
    stage =  PiPPy.PipelineStage(pipe, rank, torch.cuda.current_device())
    schedule = PiPPy.ScheduleGPipe(stage, len(placement), loss_fn = lambda x,**_: x.sum())
    with Timer() as timer:
        if rank == 0:
            schedule.step(inputs.clone())
        elif rank == world_size - 1:
            losses = []
            out = schedule.step(target = None, losses = losses)
        else:
            out = schedule.step()
    if rank == 0: print(f'Time taken by PiPPy : {timer.time():.3s}s')


def sepi(model):
    pipe = MyPipe.Pipeline(model, placement = placement, schedule = schedule) 
    sample = inputs.clone()
    with Timer() as timer:
        y, loss = pipe(sample, torch.empty(0), lambda x,**_: x.sum())
    print(f'Time taken by our framework : {timer.time():.3s}s')

if __name__ == '__main__':
    rank = os.getenv('RANK')
    local_rank = os.getenv('LOCAL_RANK')
    world_size = os.getenv('WORLD_SIZE')
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend = 'nccl')

    ...

    if dist.is_initialized():
        dist.destroy_process_group()