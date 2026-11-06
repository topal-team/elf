"""
Profile the stages of a transformer split into ``--nstages`` balanced stages, for the ILPs of the paper.

Each stage is profiled on its own with :func:`elf.profiling.profile_stage`, one at a time on the first
GPU, so that models larger than one GPU can be profiled. With two processes, the time to transfer the
output of each stage is measured between the two GPUs.

The output contains one entry per stage in the format of ``pipeline-ilps`` (``stages``) and the raw
ELF profiles (``profiles``).

Usage:
	torchrun --standalone --nproc-per-node 2 benchmarks/profile_stages.py --config-file RUNFILE \
		--nstages N --output OUTPUT
"""

import json
import argparse

import numpy as np
import torch
import torch.distributed as dist

from elf import replace_layer_with_layer_dw
from elf.profiling import profile_stage
from elf.utils import TensorMetadata, Timer
from elf.zb_utils import LayerDW

from benchmark_utils import balanced_partition, init_dist, meta_to_device
from models.utils import add_transformer_args, build_model_from_args
from paper_remat import stage_profile_to_ilp


def parse_args():
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	add_transformer_args(parser)
	parser.add_argument("--nstages", type=int, required=True)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument("--iterations", "-i", type=int, default=10)
	return parser.parse_args()


def split_stages(model, nstages):
	"""Same split as :func:`benchmark_utils.get_handcrafted_partition`, kept on the meta device."""
	factors = balanced_partition(len(model.blocks), [0] * nstages)
	stages, start = [], 0
	for s, n in enumerate(factors):
		stage = torch.nn.Sequential(*model.blocks[start : start + n])
		if s == 0:
			stage.insert(0, model.embed)
		if s == nstages - 1:
			stage.append(model.head)
		stages.append(stage)
		start += n
	return stages


def transfer_time(shape, dtype, n_iter=100):
	"""Median time (ms) to send a tensor from rank 0 to rank 1."""
	buffer = torch.empty(shape, dtype=dtype, device="cuda")
	samples = []
	for _ in range(n_iter):
		if dist.get_rank() == 0:
			with Timer() as timer:
				dist.send(buffer, 1)
			samples.append(timer.time() * 1000)
		else:
			dist.recv(buffer, 0)
	return np.median(samples).item() if samples else 0.0


def main():
	args = parse_args()
	_, rank, world_size = init_dist()
	assert world_size <= 2, "Use 1 process, or 2 to measure transfer times"

	with torch.device("meta"):
		model, dtype = build_model_from_args(args)
		replace_layer_with_layer_dw(model)

	profiles, outputs = [], []
	if rank == 0:
		metas = (TensorMetadata(model.get_sample(1, dtype=dtype, device="cuda")),)
		for s, stage in enumerate(split_stages(model, args.nstages)):
			stage = meta_to_device(stage, "cuda")
			profile, output = profile_stage(stage, metas, args.iterations)
			print(f"Stage {s}: {profile}", flush=True)
			profiles.append(profile)
			outputs.append((tuple(output.shape), output.dtype))
			metas = (TensorMetadata(output.to("cuda")),)

			stage.zero_grad(set_to_none=True)
			for module in stage.modules():
				if isinstance(module, LayerDW):
					module.clear()
			stage.to_empty(device="meta")
			torch.cuda.empty_cache()

	if world_size == 2:
		shared = [outputs]
		dist.broadcast_object_list(shared, src=0)
		outputs = shared[0]
		times = [transfer_time(shape, dt) for shape, dt in outputs[:-1]] + [0.0]
		for profile, t in zip(profiles, times):
			profile.t_transfer = t

	if rank == 0:
		with open(args.config_file, "r") as f:
			config = json.load(f)
		result = {
			"model": config["model"],
			"nparams": sum(p.numel() for p in model.parameters()),
			"stages": [stage_profile_to_ilp(p) for p in profiles],
			"profiles": [p.to_dict() for p in profiles],
		}
		with open(args.output, "w") as f:
			json.dump(result, f, indent=1)

	dist.destroy_process_group()


if __name__ == "__main__":
	main()
