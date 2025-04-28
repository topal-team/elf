import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sys
import wandb

sys.path.append("./")
from elf.zb_utils import replace_linear_with_linear_dw
import elf.pipeline as MyPipe
from elf.utils import Timer
from models.simple import SimpleTransformer

import torch.distributed.pipelining as PiPPy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pp", type=int, default=4, help="Pipeline parallelism degree")
parser.add_argument("--run_id", type=str, default="", help="Run ID")
parser.add_argument(
	"--schedule",
	type=str,
	default="zbh2",
	help="Pipeline schedule type (supported: 1f1b, zbh1, zbh2, afab)",
)
parser.add_argument("--niter", type=int, default=10, help="Number of iterations")
args = parser.parse_args()

n_stages = args.pp
n_procs = args.pp
nmb = args.pp * 2
batch_size = nmb * 2
mb_size = batch_size // nmb
seq_len = 512
model = SimpleTransformer(2000, 1024, 24, seq_len)
inputs = model.get_sample(batch_size).requires_grad_(True)
targets = model.get_target(batch_size)
loss_fn = model.loss_fn


def get_schedule(schedule_type, stage, n_stages, nmb):
	"""
	Returns the appropriate schedule object for PiPPy based on the schedule type.
	This function maps ELF schedule names to their PiPPy equivalents.

	:param schedule_type: The type of schedule to use (e.g., "1f1b", "zbh1", "zbh2")
	:param stage: PiPPy pipeline stage
	:param n_stages: Number of pipeline stages
	:param nmb_size: Number of microbatches
	:return: A PiPPy schedule object
	"""
	schedule_type = schedule_type.lower()

	if schedule_type == "1f1b":
		return PiPPy.Schedule1F1B(stage, nmb, loss_fn=loss_fn)
	elif schedule_type == "zbh1":
		return PiPPy.ScheduleInterleavedZeroBubble([stage], nmb, loss_fn=loss_fn)
	elif schedule_type == "afab":
		return PiPPy.ScheduleGPipe(stage, nmb, loss_fn=loss_fn)
	else:
		raise ValueError(f"Unknown schedule type '{schedule_type}' for PiPPy")


def get_part(rank):
	blocks_per_stage = len(model.blocks) // n_stages
	if rank == 0:
		return nn.Sequential(model.embed, *model.blocks[:blocks_per_stage]).cuda()
	elif rank == n_procs - 1:
		return nn.Sequential(*model.blocks[-blocks_per_stage:], model.head).cuda()
	else:
		return nn.Sequential(
			*model.blocks[blocks_per_stage * rank : blocks_per_stage * (rank + 1)]
		).cuda()


def pippy():
	part = get_part(rank)
	stage = PiPPy.PipelineStage(part, rank, n_stages, torch.cuda.current_device())
	schedule = get_schedule(args.schedule, stage, n_stages, nmb)
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
		for _ in range(args.niter):
			if rank == 0:
				schedule.step(inputs.clone())
			elif rank == world_size - 1:
				losses = []
				_ = schedule.step(target=targets.clone(), losses=losses)
			else:
				_ = schedule.step()

	return timer.time(), torch.cuda.max_memory_allocated() / 2**30


def elf():
	part = get_part(rank)
	replace_linear_with_linear_dw(part, "cpu")
	sources, dsts = MyPipe.get_sources_targets_sequential(list(range(world_size)))
	pipe = MyPipe.Pipeline(
		part, None, partitioner=False, schedule=args.schedule, sources=sources, targets=dsts
	)
	# Warmup
	for _ in range(5):
		y, loss = pipe(inputs.clone(), targets.clone(), loss_fn)

	torch.cuda.reset_peak_memory_stats()
	with Timer() as timer:
		for _ in range(args.niter):
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

	dist.barrier()
	torch.cuda.empty_cache()
	torch.cuda.synchronize()

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
		config_dict = {
			"pp_size": args.pp,
			"batch_size": batch_size,
			"seq_len": seq_len,
			"schedule": args.schedule,
		}

		wandb.init(
			project="compare-frameworks",
			entity="topal-inria",
			job_type="framework-comparison",
			config=config_dict,
			mode="offline",
		)

		# Log metrics
		metrics = {
			"run_id": args.run_id,
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
