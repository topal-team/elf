import os
import sys
import wandb
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.append("./")

import elf
import torch.distributed.pipelining as PiPPy

from elf.utils import Timer
from models.utils import add_transformer_args, build_model_from_args
from benchmarks.benchmark_utils import meta_to_device

parser = argparse.ArgumentParser()

# Model hyper-parameters
add_transformer_args(parser, model_type="full")

# Script-specific arguments
parser.add_argument("--pp", type=int, default=4, help="Pipeline parallelism degree")
parser.add_argument("--run-id", type=str, default="", help="Run ID")
parser.add_argument(
	"--schedule", type=str, help="Pipeline schedule type (supported: afab, 1f1b, megatron, zbh1, zbv)"
)
parser.add_argument("--mb-size", type=int, default=2, help="Microbatch size")
parser.add_argument("--niters", type=int, default=10, help="Number of iterations")
parser.add_argument(
	"--only",
	type=str,
	choices=["torch", "elf", "both"],
	default="both",
	help="Run only specified framework benchmark (torch, elf, or both)",
)

args = parser.parse_args()

nmb = args.pp * 2
batch_size = args.mb_size * nmb
with torch.device("meta"):
	model = build_model_from_args(args, model_type="full")

inputs = model.get_sample(batch_size)
targets = model.get_target(batch_size)
loss_fn = model.loss_fn

assert args.nblocks >= args.pp, (
	"Number of blocks must be greater than or equal to pipeline parallelism degree"
)


# TODO: use elf.Placement.default
def get_placement(schedule_type, world_size):
	match schedule_type:
		case "1f1b" | "zbh1" | "afab" | "zbh2":
			placement = list(range(world_size))
		case "megatron":
			placement = list(range(world_size)) * 2
		case "zbv":
			placement = list(range(world_size)) + list(reversed(range(world_size)))
		case _:
			raise ValueError(f"Unknown schedule type '{args.schedule}'")

	return placement


def get_schedule(schedule_type, stages, nmb):
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

	match schedule_type:
		case "afab":
			assert len(stages) == 1
			return PiPPy.ScheduleGPipe(stages[0], nmb, loss_fn=loss_fn)
		case "1f1b":
			assert len(stages) == 1
			return PiPPy.Schedule1F1B(stages[0], nmb, loss_fn=loss_fn)
		case "megatron":
			assert len(stages) == 2
			return PiPPy.ScheduleInterleaved1F1B(stages, nmb, loss_fn=loss_fn)
		case "zbh1":
			assert len(stages) == 1
			return PiPPy.ScheduleInterleavedZeroBubble(stages, nmb, loss_fn=loss_fn)
		case "zbv":
			assert len(stages) == 2
			return PiPPy.ScheduleZBVZeroBubble(stages, nmb, loss_fn=loss_fn)
		case _:
			raise ValueError(f"Unknown schedule type '{schedule_type}' for PiPPy")


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
			parts.append(nn.Sequential(model.embed, *model.blocks[start:end]))
		elif i == len(placement) - 1:
			parts.append(nn.Sequential(*model.blocks[start:end], model.head))
		else:
			parts.append(nn.Sequential(*model.blocks[start:end]))

		start = end

	return [meta_to_device(p) for p in parts]


def find_stage_global_idx(rank, placement, local_idx):
	cpt = 0
	for i, p in enumerate(placement):
		if p == rank:
			if cpt == local_idx:
				return i

			cpt += 1

	return None


def benchmark_pippy():
	placement = get_placement(args.schedule, world_size)
	parts = get_parts(rank, placement)
	n_stages = len(placement)
	stages = [
		PiPPy.PipelineStage(
			p, find_stage_global_idx(rank, placement, i), n_stages, torch.cuda.current_device()
		)
		for i, p in enumerate(parts)
	]
	schedule = get_schedule(args.schedule, stages, nmb)

	def get_args_kwargs():
		input_args = []
		input_kwargs = {}
		if rank == placement[0]:
			input_args.append(inputs.clone())
		if rank == placement[-1]:
			input_kwargs["target"] = targets.clone()

		return input_args, input_kwargs

	# Warmup
	for _ in range(5):
		input_args, input_kwargs = get_args_kwargs()
		_ = schedule.step(*input_args, **input_kwargs)

	torch.cuda.reset_peak_memory_stats()
	with Timer() as timer:
		for _ in range(args.niters):
			input_args, input_kwargs = get_args_kwargs()
			_ = schedule.step(*input_args, **input_kwargs)

	return timer.time(), torch.cuda.max_memory_allocated() / 2**30


def benchmark_elf():
	placement = get_placement(args.schedule, world_size)
	parts = get_parts(rank, placement)

	scheduler = args.schedule
	if scheduler == "megatron":
		scheduler = "1f1b"  # Not the same name

	for part in parts:
		elf.replace_linear_with_linear_dw(part, "cpu")

	sources, dsts = elf.get_sources_targets_sequential(placement)
	pipe = elf.Pipeline(
		parts,
		None,
		partitioner=False,
		scheduler=scheduler,
		placement=placement,
		sources=sources,
		targets=dsts,
	)
	# Warmup
	for _ in range(5):
		y, loss = pipe(inputs.clone(), targets.clone(), loss_fn, split_size=args.mb_size)

	torch.cuda.reset_peak_memory_stats()
	with Timer() as timer:
		for _ in range(args.niters):
			y, loss = pipe(inputs.clone(), targets.clone(), loss_fn, split_size=args.mb_size)

	return timer.time(), torch.cuda.max_memory_allocated() / 2**30


if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	world_size = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl", device_id=torch.device(local_rank))

	inputs = inputs.cuda()
	targets = targets.cuda()

	if args.only == "elf":
		torch_time, torch_mem = (0.0, 0.0)
	else:
		torch_time, torch_mem = benchmark_pippy()

	dist.barrier()
	torch.cuda.empty_cache()
	torch.cuda.synchronize()

	if args.only == "torch":
		elf_time, elf_mem = (0.0, 0.0)
	else:
		elf_time, elf_mem = benchmark_elf()

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
			"seq_len": args.seq_len,
			"schedule": args.schedule,
			"niters": args.niters,
			"nblocks": args.nblocks,
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

	del model
	del inputs
	del targets

	if dist.is_initialized():
		dist.destroy_process_group()
