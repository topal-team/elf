import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.models import resnet50, resnet18
from pipeline import Pipeline
import os

"""
Start this file with torchrun
"""
GROUPS_STORAGE='/net/home/project/tutorial/tutorial050/topal-internship/helios/'
if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")
	print('RANKS:\t', dist.get_rank(), rank, local_rank)

	model = resnet18()
	if rank==0:
		print(dict(model.named_parameters()).keys())
	sample = torch.randn((32, 3, 224, 224)).cuda()
	pipe = Pipeline(model, sample)
	if rank==0:
		print('PIPE PARAMS:\n', pipe.named_parameters().keys())
	pipe.init_optimizer()
	loss_fn = nn.functional.cross_entropy
	for e in range(10):
		if rank == 0:
			print(f"Epoch {e}")
		sample = torch.randn((32, 3, 224, 224)).cuda()
		target = torch.randn((32, 1000)).cuda()
		# _ = model(sample, target, loss_fn, profile=True)
		# optimizer.step()
		_ = pipe(sample, target, loss_fn, profile=False)
		pipe.optimizer.step()
		pipe.save_state_dict(e, f'${GROUPS_STORAGE}/checkpoints')
	pipe.clear()

	if dist.is_initialized():
		dist.destroy_process_group()
