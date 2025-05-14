import os
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.profiler as profiler
import torch.distributed.pipelining as pipelining

from torch.cuda import cudart

sys.path.append(".")
from models.simple import SimpleTransformer


def get_parts(rank, placement):
	parts = []
	blocks_per_stage = len(model.blocks) // len(placement)
	start, end = 0, 0
	for i, p in enumerate(placement):
		end += blocks_per_stage + (1 if i < (len(model.blocks) % len(placement)) else 0)
		if rank != p:
			start = end
			continue

		if i == 0:
			parts.append(nn.Sequential(model.embed, *model.blocks[start:end]).cuda())
		elif i == len(placement) - 1:
			parts.append(nn.Sequential(*model.blocks[start:end], model.head).cuda())
		else:
			parts.append(nn.Sequential(*model.blocks[start:end]).cuda())

		start = end

	return parts


def find_stage_global_idx(rank, placement, local_idx):
	cpt = 0
	for i, p in enumerate(placement):
		if p == rank:
			if cpt == local_idx:
				return i

			cpt += 1

	return None


if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	ws = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	# Create model and sample data
	model = SimpleTransformer(2000, 1024, 6 * ws, 512)
	nmb = ws

	placement = list(range(ws))
	parts = get_parts(rank, placement)
	stage_idx = find_stage_global_idx(rank, placement, 0)

	n_stages = len(placement)
	stages = [
		pipelining.PipelineStage(
			p, find_stage_global_idx(rank, placement, i), n_stages, torch.cuda.current_device()
		)
		for i, p in enumerate(parts)
	]

	# schedule = pipelining.ScheduleInterleavedZeroBubble(stages, nmb, loss_fn=model.loss_fn)
	schedule = pipelining.Schedule1F1B(stage=stages[0], n_microbatches=nmb, loss_fn=model.loss_fn)
	optimizer = torch.optim.Adam(model.parameters())
	inputs = model.get_sample(32).cuda()
	targets = model.get_target(32).cuda()

	def get_args_kwargs():
		input_args = []
		input_kwargs = {}
		if rank == placement[0]:
			input_args.append(inputs.clone())
		if rank == placement[-1]:
			input_kwargs["target"] = targets.clone()

		return input_args, input_kwargs

	# Warmup
	for _ in range(3):
		input_args, input_kwargs = get_args_kwargs()
		_ = schedule.step(*input_args, **input_kwargs)
		optimizer.step()

	torch.cuda.synchronize()

	profiler.start()
	cudart().cudaProfilerStart()

	for i in range(5):
		if rank == 0:
			print(f"Iteration {i}")
		_ = schedule.step(*input_args, **input_kwargs)
		optimizer.step()

	cudart().cudaProfilerStop()
	profiler.stop()

	if dist.is_initialized():
		dist.destroy_process_group()
