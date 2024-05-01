import os
import torch
import torch.distributed as dist
from pipeline.pipeline import Pipeline
from settings import model, inputs
from argparse import ArgumentParser

import logging
logger = logging.getLogger('profiler')
logging.basicConfig(level = logging.INFO)

if __name__ == "__main__":
    parser = ArgumentParser(description = "Profile forward/backward pass of pipeline on GPT")
    parser.add_argument('--log', choices=['debug', 'info', 'none'], default='info', required=False, help="logging level")
    args = parser.parse_args()
    match args.log:
        case 'debug':
            logging.getLogger().setLevel(logging.DEBUG)
        case 'info':
            logging.getLogger().setLevel(logging.INFO)
        case 'none':
            logging.getLogger().setLevel(100)

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK")))
    dist.init_process_group(backend = "nccl")

    pipe = Pipeline(model, placement = [0, 1, 2, 3], schedule = "afab")
    # Warmup
    for _ in range(5):
        pipe(inputs.cuda(), torch.empty(0), lambda x,y,**_: x.sum(), split_size=2, profile = None)
    
    torch.cuda.cudart().cudaProfilerStart()

    pipe(inputs.cuda(), torch.empty(0), lambda x,y,**_: x.sum(), split_size=2, profile = None)

    torch.cuda.cudart().cudaProfilerStop()

    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
