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
    
    pipelined = PipelineGPT(model, devices = placement, vocab_size = vocab_size, input_shape = inputs.shape)

    sizes = [1, 2, 4, 8, 16, 32]
    times = []
    for block_size in sizes:
        pipelined = Pipe(pipelined, chunks = vocab_size // block_size)
    
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
        print(f'Time taken by torch pipe (block size = {block_size}) : {end - start:.2f}s. Average : {t:.3f}s')
        times.append(t)
        with open(f'torch_GPTXXXL_{dataset_size}.txt', 'w') as file:
            for size, t in zip(sizes, times):
                file.write(f'{size}, {t}\n')
