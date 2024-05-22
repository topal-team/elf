import torch
import torch.distributed as dist
import os
import time
import numpy as np
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
    torch.cuda.cudart().cudaProfilerStart()
    
    inputs = inputs.cuda()

    times = []
    for size in split_sizes:
        #if (batch_size // size) % len(placement) != 0:
            # if global_rank == 0: logger.warning(f'The number of micro batches should be a multiple of the number of stages ! Got {batch_size // size} and {len(placement)}. Skipping.')
            # continue
        if global_rank == 0: logger.info(f'Beginning bench for micro batches of size {size}')

        pipe = Pipeline(model, placement, schedule = schedule)
        
        # Warmup
        for i in range(3):
            if global_rank == 0: logger.info(f'Warmup {i}')
            _ = pipe(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), size)
        torch.cuda.reset_peak_memory_stats()
        
        iter_times = []
        idles = []
        for i in range(iters):
            start = time.time()
            _ = pipe(inputs.clone(), torch.empty(0), lambda x,y,**_: x.sum(), size)
            end = time.time()
            iter_times.append(end - start)
            idles.append(pipe.engine.idle_time / pipe.engine.total_time)
            
        t = sorted(iter_times)[iters // 2] # median
        i = sorted(idles)[iters // 2] # median
        
        std = np.std(iter_times)
        if std > (t / 20):
            logger.warning(f'Very high standard deviation for iter times ! ({std} for t = {t}) Something is probably wrong')
        std = np.std(idles)
        if std > (i / 20):
            logger.warning(f'Very high standard deviation for idle times ! ({std} for i = {i}) Something is probably wrong')
        
        mems = [torch.tensor(0.0, device = rank) for _ in range(world_size)] if global_rank == 0 else None
        dist.gather(torch.tensor(torch.cuda.max_memory_allocated() / (2**30), device = rank), mems, 0)
        
        itimes = [torch.tensor(0.0, device = rank) for _ in range(world_size)] if global_rank == 0 else None
        dist.gather(torch.tensor(i, device = rank), itimes, 0)
        
        if global_rank == 0:
            print(f'Size {size} :\n\tMedian time : {t:.3f}s\n\tIdle time : {i:.3f}s or {100 * i / t:.1f}% of total time')
            
            f.write(f'{size},{t}')
            for i in itimes:
                f.write(f',{i}')
            for m in mems:
                f.write(f',{m}')
            f.write('\n')
        times.append(t)

    torch.cuda.cudart().cudaProfilerStop()
    
    if global_rank == 0: f.close()
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
