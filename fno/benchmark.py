#!/usr/bin/env python
"""
Benchmarking script for ILPS-guided FNO model execution strategies.

This script benchmarks FNO model performance using different execution strategies
(balanced partitioning, recomputation, etc.) based on ILP solver solutions. It measures
execution time and peak memory usage for each strategy and reports the results.

Usage:
    torchrun --nproc-per-node=NUM_GPUS fno/benchmark.py \
        --config-file CONFIG_FILE --solution-file SOLUTION_FILE \
        --output-file OUTPUT_FILE --solution-type SOLUTION_TYPE [options]
"""

import os
import sys
import json
import argparse
import logging
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn

from datetime import timedelta
from typing import Dict, List, Optional, Any

from models.simple import SimpleFNO, SimpleTFNO
from benchmarks.benchmark_utils import bench, meta_to_device, balanced_partition
from benchmarks.ilp_schedulers import RematScheduler
from neuralop.layers.complex import ctanh
from elf import replace_layer_with_layer_dw

logging.basicConfig(level=logging.INFO)


def setup_logging(log_level: str) -> None:
	"""Configure logging level based on command line argument."""
	match log_level:
		case "info":
			logging.getLogger().setLevel(logging.INFO)
		case "debug":
			logging.getLogger().setLevel(logging.DEBUG)
		case "none":
			logging.getLogger().setLevel(100)


def setup_distributed() -> tuple[int, int]:
	"""Initialize distributed training environment."""
	local_rank = int(os.getenv("LOCAL_RANK"))
	torch.cuda.set_device(local_rank)
	timeout = 300
	os.environ["ELF_TIMEOUT"] = str(timeout)
	dist.init_process_group(
		backend="nccl", device_id=torch.device(f"cuda:{local_rank}"), timeout=timedelta(seconds=timeout)
	)
	return dist.get_rank(), dist.get_world_size()


def load_results(output_file: str, restart: bool) -> Dict:
	"""Load existing results or create new results dictionary."""
	if restart:
		return {}
	try:
		with open(output_file, "r") as f:
			return json.load(f)
	except (FileNotFoundError, json.JSONDecodeError):
		return {}


class FNOBlock(nn.Module):
	"""
	FNO block implementation that does not register all submodules, but only the ones that are part of the current block.
	Uses the postactivation version.
	"""

	def __init__(self, root, index: int):
		super().__init__()

		self.fno_skips = root.fno_skips[index]
		self.convs = root.convs[index]
		self.channel_mlp_skips = root.channel_mlp_skips[index]
		self.channel_mlp = root.channel_mlp[index]
		self.norm = root.norm
		self.non_linearity = root.non_linearity
		self.complex_data = root.complex_data
		self.use_channel_mlp = root.use_channel_mlp
		self.stabilizer = root.stabilizer

	def forward(self, x, output_shape=None):
		if self.fno_skips is not None:
			x_skip_fno = self.fno_skips(x)
			x_skip_fno = self.convs.transform(x_skip_fno, output_shape=output_shape)

		if self.use_channel_mlp and self.channel_mlp_skips is not None:
			x_skip_channel_mlp = self.channel_mlp_skips(x)
			x_skip_channel_mlp = self.convs.transform(x_skip_channel_mlp, output_shape=output_shape)

		if self.stabilizer == "tanh":
			if self.complex_data:
				x = ctanh(x)
			else:
				x = torch.tanh(x)

		x_fno = self.convs(x, output_shape=output_shape)

		if self.norm is not None:
			x_fno = self.norm(x_fno)

		x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

		x = self.non_linearity(x)

		if self.use_channel_mlp:
			if self.channel_mlp_skips is not None:
				x = self.channel_mlp(x) + x_skip_channel_mlp
			else:
				x = self.channel_mlp(x)

		if self.norm is not None:
			x = self.norm(x)

		x = self.non_linearity(x)

		return x


def build_fno_stages(model: nn.Module, factors: List[int]) -> List[nn.Sequential]:
	"""Build all FNO stages from a model and per-stage block counts."""
	n_blocks = model.fno_blocks.n_layers
	num_stages = len(factors)
	assert int(sum(factors)) == int(n_blocks), (
		f"Sum of factors ({sum(factors)}) != number of blocks ({n_blocks})"
	)

	stages = []
	start_idx = 0
	for i in range(num_stages):
		modules = []

		if i == 0:
			modules.append(model.positional_embedding)
			modules.append(model.lifting)

		for j in range(start_idx, start_idx + factors[i]):
			modules.append(FNOBlock(model.fno_blocks, j))

		if i == num_stages - 1:
			modules.append(model.projection)

		stages.append(nn.Sequential(*modules))
		start_idx += factors[i]

	return stages


def partition_fno(model: nn.Module, rank: int, placement: List[int], factors: List[int]):
	"""Partition FNO model for a specific rank with given factors."""
	stages = build_fno_stages(model, factors)
	parts = [stages[i] for i, p in enumerate(placement) if p == rank]
	parts = [meta_to_device(p, "cuda") for p in parts]
	return parts


def build_fno_from_config(config: Dict) -> tuple[SimpleFNO, torch.dtype]:
	"""Build FNO model from config dictionary."""
	model_cfg = config["model"]
	data_cfg = config.get("data", {})
	complex_data = data_cfg.get("complex", True)
	data = config["data"]
	spatial_dims = tuple(data["spatial_dims"])
	arch_mapping = {"fno": SimpleFNO, "tfno": SimpleTFNO}
	arch = model_cfg.get("architecture").lower()

	model = arch_mapping[arch](
		spatial_dims,
		n_modes=tuple(model_cfg["n_modes"]),
		in_channels=model_cfg["in_channels"],
		out_channels=model_cfg["out_channels"],
		hidden_channels=model_cfg["hidden_channels"],
		n_layers=model_cfg["n_layers"],
		projection_channel_ratio=model_cfg.get("projection_channel_ratio"),
		complex_data=complex_data,
	)

	dtype = torch.complex64 if complex_data else torch.float32
	return model, dtype


def log_model_info(model: nn.Module, rank: int) -> None:
	"""Log model information."""
	if rank == 0:
		print(f"\nModel has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
		allocated = torch.cuda.memory_allocated() / (1024**3)
		free = torch.cuda.get_device_properties().total_memory / (1024**3) - allocated
		print(f"Memory allocated: {allocated:.2f}GB")
		print(f"Memory free: {free:.2f}GB")


def run_benchmark(
	model: nn.Module,
	parts: Any,
	scheduler: str,
	placement: List[int],
	nmb: int,
	dtype: torch.dtype,
	rank: int = 0,
	ntrials: int = 1,
	stats_file: Optional[str] = None,
) -> tuple[float, List[float], List[float]]:
	"""Run benchmark and return iteration time and peak memory usage."""
	iter_time, all_peak_mems, times = bench(
		model, parts, scheduler, placement, dtype, nmb, ntrials, stats_file=stats_file
	)
	if rank == 0:
		print(f"\t{iter_time:.2f}s / iter, Peak memory: {[f'{m:.2f}' for m in all_peak_mems]} GB")
	return iter_time, all_peak_mems, times


def process_solution(
	model: nn.Module,
	solution: Optional[Dict],
	solution_type: str,
	nmb: int,
	rank: int,
	dtype: torch.dtype,
	ntrials: int,
	stats_file: Optional[str] = None,
) -> tuple[Optional[float], Optional[List[float]], Optional[List[float]]]:
	"""Process a single solution type and return benchmark results."""

	placement = solution.get("placement")
	# For FNO, we need to partition the fno_blocks, not model.blocks
	balance = balanced_partition(model.fno_blocks.n_layers, placement)

	if solution is None:
		if rank == 0:
			print(f"No {solution_type} solution found")
		return None, None, None

	if rank == 0:
		print(f"{solution_type}:")

	scheduler = RematScheduler(solution)
	parts = partition_fno(model, rank, placement, balance)
	return run_benchmark(
		model,
		parts,
		scheduler,
		placement,
		nmb,
		dtype,
		rank=rank,
		ntrials=ntrials,
		stats_file=stats_file,
	)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--log", choices=["info", "debug", "none"], default="none")
	parser.add_argument(
		"--solution-file", type=str, required=True, help="Path to the ILP solution file"
	)
	parser.add_argument(
		"--output-file",
		type=str,
		default="results/fno-bench-ilps-default.json",
		help="Path to the output file",
	)
	parser.add_argument(
		"--solution-type", type=str, required=True, help="Specific solution type to benchmark"
	)
	parser.add_argument(
		"--config-file",
		type=str,
		required=True,
		help="Path to the config file with model hyperparameters",
	)
	parser.add_argument(
		"--detailed-stats-file",
		type=str,
		default=None,
		help="Path to write per-rank detailed memory stats (requires ELF_MEMORY=1)",
	)
	args = parser.parse_args()
	setup_logging(args.log)

	rank, world_size = setup_distributed()

	# Ensure output directory exists
	if rank == 0:
		output_dir = os.path.dirname(args.output_file)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)

	# Load solutions and config
	with open(args.solution_file, "r") as f:
		solution_data = json.load(f)
		solutions = solution_data.get("solutions", {})

	with open(args.config_file, "r") as f:
		config = json.load(f)

	# Create and initialize model
	with torch.device("meta"):
		model, dtype = build_fno_from_config(config)
		replace_layer_with_layer_dw(model)

	log_model_info(model, rank)

	if rank == 0:
		gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
		process = psutil.Process(os.getpid())
		cpu_allocated = process.memory_info().rss / (1024**3)
		print(
			f"Memory allocated before {args.solution_type}: GPU {gpu_allocated:.2f}GB, CPU {cpu_allocated:.2f}GB"
		)

	solution = solutions.get(args.solution_type)
	nmb = config.get("nmb", None)
	ntrials = config.get("ntrials", 5)

	try:
		iter_time, peak_mems, all_times = process_solution(
			model,
			solution,
			args.solution_type,
			nmb,
			rank,
			dtype,
			ntrials,
			stats_file=args.detailed_stats_file,
		)

		if rank == 0 and iter_time is not None:
			results = {
				"time": iter_time,
				"peak_mems": [float(m) for m in peak_mems],
				"all_times": all_times,
				"objective": solution.get("objective"),
			}

			# Save results atomically
			with open(args.output_file, "w") as f:
				json.dump(results, f, indent=4)

	except torch.cuda.OutOfMemoryError as e:
		print(f"Out of memory on rank {rank} for {args.solution_type}")
		print(f"Error: {e}")
		if rank == 0:
			with open(args.output_file, "w") as f:
				json.dump({"error": f"Out of memory (rank {rank})"}, f, indent=4)

		sys.exit(1)

	except Exception as e:
		import traceback

		print(f"Error on rank {rank} processing {args.solution_type}: {str(e)}")
		print("Full stacktrace:")
		traceback.print_exc()
		sys.exit(1)

	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
