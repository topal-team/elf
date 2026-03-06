import os


import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.models import resnet50

from elf import Pipeline, sequential_signatures
from elf.zb_utils import replace_linear_with_linear_dw

import logging

logging.basicConfig(level=logging.INFO)


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
	dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

	model = resnet50()
	replace_linear_with_linear_dw(model, rank)
	part = get_part(model, rank)

	placement = [0, 1, 2, 3]

	signatures = sequential_signatures(placement)
	# This is equivalent to the above, but more explicit
	# sources = {0: {"input": None}, 1: {"input": 0}, 2: {"input": 1}, 3: {"input": 2}}
	# targets = {0: {"output": [1]}, 1: {"output": [2]}, 2: {"output": [3]}, 3: {"output": [None]}}
	# signatures = signatures_from_sources_targets(sources, targets)

	# We don't need to pass a sample here, because no profiling will be done
	pipe = Pipeline(
		part, None, partitioner=False, placement=placement, signatures=signatures, scheduler="zbh2"
	)

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

	dist.destroy_process_group()
