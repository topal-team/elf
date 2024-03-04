import torch
import torch.nn as nn
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet34()
inputs = torch.randn(4, 3, 224, 224)

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, use_cuda=True) as prof:
    with record_function("inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
