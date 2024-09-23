import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.models import resnet50
from pipeline import Pipeline
import os

"""
Start this file with torchrun
"""

if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	model = resnet50()
	sample = torch.randn((32, 3, 224, 224)).cuda()
	model = Pipeline(model, sample)
	loss_fn = nn.functional.cross_entropy
	optimizer = torch.optim.Adam(model.parameters())
	for e in range(10):
		if rank == 0:
			print(f"Epoch {e}")
		sample = torch.randn((32, 3, 224, 224)).cuda()
		target = torch.randn((32, 1000)).cuda()
		_ = model(sample, target, loss_fn)
		optimizer.step()

	model.clear()

	if dist.is_initialized():
		dist.destroy_process_group()
