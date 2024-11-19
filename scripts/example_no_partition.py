import os
import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.models import resnet50

from pipeline import Pipeline


def get_part(model, rank):
	if rank == 0:
		return nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1)
	elif rank == 1:
		return model.layer2
	elif rank == 2:
		return model.layer3
	else:
		return nn.Sequential(model.layer4, model.avgpool, nn.Flatten(1), model.fc)


if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	model = resnet50()
	part = get_part(model, rank)

	# We don't need a sample here
	pipe = Pipeline(part, None, partition=False)

	loss_fn = nn.functional.cross_entropy
	optimizer = torch.optim.Adam(pipe.parameters())

	for e in range(10):
		if rank == 0:
			print(f"Sample {e}")

		sample = torch.randn((32, 3, 224, 224)).cuda()
		target = torch.randn((32, 1000)).cuda()
		optimizer.zero_grad()
		y, loss = pipe(sample, target, loss_fn)
		optimizer.step()

	pipe.clear()
