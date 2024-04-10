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

    split_sizes = [1, 2, 4, 8, 16, 32, 64]
    times = []
    for size in split_sizes:
        pipelined = Pipe(sequential, chunks = batch_size // size, checkpoint = 'never')
    
        # Warmup
        for _ in range(5):
            y = pipelined(inputs.clone()).local_value()[0]
            loss = y.sum()
            loss.backward()
        start = time.time()
        for _ in range(iters):
            y = pipelined(inputs.clone()).local_value()[0]
            loss = y.sum()
            loss.backward()
        end = time.time()
        t = (end - start) / iters
        f.write(f'{size},{t}\n')
        print(f'Time taken by torch pipe (size {size}) : {end - start:.2f}s. Average : {t:.3f}s')
        times.append(t)
