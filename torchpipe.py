from models.GPT import PipelineGPT
import torch.distributed as dist
from torch.distributed.pipeline.sync import Pipe
import os
import time
from settings import *

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29502'
    dist.rpc.init_rpc('worker', rank = 0, world_size = 1)
        
    inputs = inputs.cuda()
    
    sequential = PipelineGPT(model, devices = placement, vocab_size = vocab_size, input_shape = inputs.shape)

    f = open("torchpipe.out", "w")

    split_sizes = [1, 2, 4, 8, 16, 32]
    times = []
    for size in split_sizes:
        pipelined = Pipe(sequential, chunks = batch_size // size, checkpoint = 'never')
    
        # Warmup
        for _ in range(3):
            y = pipelined(inputs.clone()).local_value()[0]
            loss = y.sum()
            loss.backward()

        torch.cuda.reset_peak_memory_stats()
        iter_times = []
        for _ in range(iters):
            start = time.time()
            y = pipelined(inputs.clone()).local_value()[0]
            loss = y.sum()
            loss.backward()
            end = time.time()
            iter_times.append(end - start)
        t = sorted(iter_times)[iters // 2] # median
        f.write(f'{size},{t}\n')
        print(f'Time taken by torch pipe (size {size}) : {end - start:.2f}s. Median : {t:.3f}s')
        times.append(t)
