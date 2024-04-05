from models.GPT import GPT, PipelineGPT
from pipeline.pipeline import create_pipeline
from pipeline.schedule import generate_afab_schedule
from pipeline.engine import StageScheduler
from argparse import ArgumentParser
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipeline.sync import Pipe
import os
import time
from settings import *

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29502'
    dist.rpc.init_rpc('worker', rank=0, world_size=1)
        
    inputs = inputs.cuda()
    
    pipelined = PipelineGPT(model, devices = placement, vocab_size = vocab_size, input_shape = inputs.shape)
    pipelined = Pipe(pipelined)
    
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
    print(f'Time taken by torch pipe : {end - start:.2f}s. Average : {(end - start) / iters:.3f}s')
    # Time taken by torch pipe : 14.91s. Average : 0.298s (GPT XXXL)
    # Time taken by torch pipe : 3.75s. Average : 0.075s (ResNet Bottleneck 8)
