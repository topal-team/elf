import gc
import time
import torch
import torch.nn as nn
import psutil
from utils import activations_offloading

def print_memory_usage(stage):
    gc.collect()
    torch.cuda.empty_cache()
    process = psutil.Process()
    print(f"Memory usage {stage}:")
    print(f"  CPU memory: {process.memory_info().rss / 1024 ** 3:.2f} GB")
    print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    # print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

torch.cuda.cudart().cudaProfilerStart()

model = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),
                      nn.Conv2d(32, 64, 3, padding=1),
                      nn.Conv2d(64, 128, 3, padding=1),
                      nn.Conv2d(128, 256, 3, padding=1),
                      nn.Conv2d(256, 1, 1),
                      nn.Flatten(),
                      nn.Linear(1024**2, 1)).cuda()

inputs = torch.randn((4, 1, 1024, 1024), device = 'cuda')
targets = torch.randn((4, 1), device = 'cuda')


print_memory_usage('before forward')

with activations_offloading():
    y = model(inputs)
print_memory_usage('after forward')

activations_offloading().wait_for_offloading()
print_memory_usage('end of offloading')

activations_offloading().prefetch()
print_memory_usage('starting to prefetch')

loss = torch.nn.functional.mse_loss(y, targets)
loss.backward()
print_memory_usage('after backward')

torch.cuda.cudart().cudaProfilerStop()
