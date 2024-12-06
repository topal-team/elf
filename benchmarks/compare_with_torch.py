import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import subprocess
import wandb

sys.path.append("./")
import elf.pipeline as MyPipe
from elf.utils import Timer
from models.simple import SimpleTransformer

import torch.distributed.pipelining as PiPPy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pp", type=int, default=4, help="Pipeline parallelism degree")
args = parser.parse_args()

n_stages = args.pp
n_procs = args.pp
batch_size = 32
seq_len = 512
model = SimpleTransformer(4000, 2048, 8 * args.pp, 512)
inputs = model.get_sample(batch_size)
targets = model.get_target(batch_size)
loss_fn = model.loss_fn
mb_size = batch_size // n_procs


def get_part(rank):
	blocks_per_stage = len(model.blocks) // n_stages
	if rank == 0:
		return nn.Sequential(model.embed, *model.blocks[:blocks_per_stage]).cuda(), inputs.clone()[
			:mb_size
		].cuda()
	elif rank == n_procs - 1:
		return nn.Sequential(*model.blocks[-blocks_per_stage:], model.head).cuda(), torch.randn(
			mb_size, seq_len, model.hidden_dim
		).cuda()
	else:
		return nn.Sequential(
			*model.blocks[blocks_per_stage * rank : blocks_per_stage * (rank + 1)]
		).cuda(), torch.randn(mb_size, seq_len, model.hidden_dim).cuda()


def pippy():
	part, sample = get_part(rank)
	stage = PiPPy.PipelineStage(part, rank, n_stages, torch.cuda.current_device())
	schedule = PiPPy.Schedule1F1B(stage, n_stages, loss_fn=loss_fn)
	# Warmup
	for _ in range(5):
		if rank == 0:
			schedule.step(inputs.clone())
		elif rank == world_size - 1:
			losses = []
			_ = schedule.step(target=targets.clone(), losses=losses)
		else:
			_ = schedule.step()

	torch.cuda.reset_peak_memory_stats()
	with Timer() as timer:
		for _ in range(10):
			if rank == 0:
				schedule.step(inputs.clone())
			elif rank == world_size - 1:
				losses = []
				_ = schedule.step(target=targets.clone(), losses=losses)
			else:
				_ = schedule.step()

	return timer.time(), torch.cuda.max_memory_allocated() / 2**30


def elf():
	pipe = MyPipe.Pipeline(model, inputs.clone(), schedule="1f1b")
	# Warmup
	for _ in range(5):
		y, loss = pipe(inputs.clone(), targets.clone(), loss_fn)

	torch.cuda.reset_peak_memory_stats()
	with Timer() as timer:
		for _ in range(10):
			y, loss = pipe(inputs.clone(), targets.clone(), loss_fn)

	return timer.time(), torch.cuda.max_memory_allocated() / 2**30


if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	world_size = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	inputs = inputs.cuda()
	targets = targets.cuda()

	torch_time, torch_mem = pippy()
	elf_time, elf_mem = elf()

	# Gather memory stats from all GPUs
	elf_mems = (
		[torch.tensor(0.0, device=local_rank) for _ in range(world_size)] if rank == 0 else None
	)
	torch_mems = (
		[torch.tensor(0.0, device=local_rank) for _ in range(world_size)] if rank == 0 else None
	)

	dist.gather(torch.tensor(elf_mem, device=local_rank), elf_mems, 0)
	dist.gather(torch.tensor(torch_mem, device=local_rank), torch_mems, 0)

	if rank == 0:
		# Log config
		config_dict = {"pp_size": args.pp, "batch_size": batch_size, "seq_len": seq_len}

		# Get git commit hash
		try:
			git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
			config_dict["git_commit"] = git_commit
		except (subprocess.CalledProcessError, FileNotFoundError):
			config_dict["git_commit"] = "unknown"

		wandb.init(
			project="compare-frameworks",
			entity="topal-inria",
			job_type="framework-comparison",
			config=config_dict,
			mode="offline",
		)

		# Log metrics
		metrics = {
			"pp_size": args.pp,
			"parameters": sum(p.numel() for p in model.parameters()),
			"torch_time": torch_time,
			"elf_time": elf_time,
		}

		# Log memory for each GPU
		for i, (elf_m, torch_m) in enumerate(zip(elf_mems, torch_mems)):
			metrics[f"gpu_{i}_elf_memory_gb"] = elf_m.item()
			metrics[f"gpu_{i}_torch_memory_gb"] = torch_m.item()

		wandb.log(metrics)
		wandb.finish()

	if dist.is_initialized():
		dist.destroy_process_group()
