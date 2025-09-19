#!/usr/bin/env python
"""
Memory comparison benchmark script for comparing ILP memory estimates with actual execution.

This script runs a benchmark with ELF_MEMORY=1 to collect detailed memory statistics
during execution and compares them with the ILP memory estimates from stage_remat.py.
It generates both the raw data and optional plots for visualization.

Usage:
	torchrun --nproc-per-node=NUM_GPUS benchmarks/memory_comparison_benchmark.py --solution-file SOLUTION_FILE --config-file CONFIG_FILE --output-file OUTPUT_FILE --solution-type SOLUTION_TYPE [options]

Arguments:
	--solution-file: Path to the ILP solution file
	--config-file: Path to the ILP config file
	--output-file: Path to write comparison results
	--solution-type: Solution type to analyze (e.g., "StageRemat", "Unbalancing", "BlockRemat")
	--plot: Generate plots (saves to same directory as output file)
	--log: Logging level (info, debug, or none)
"""

import os
import sys
import json
import datetime
import argparse
import logging
from typing import Dict, List, Any

import torch
import torch.distributed as dist

# Set ELF_MEMORY=1 to enable detailed memory tracking
os.environ["ELF_MEMORY"] = "1"

from elf import replace_linear_with_linear_dw, Pipeline, get_sources_targets_sequential
from models.simple import FullTransformer
from models.utils import add_transformer_args, model_config_from_args
from benchmarks.benchmark_utils import get_handcrafted_imbalanced_partition, balanced_partition
from benchmarks.ilp_schedulers import RematScheduler

# Import memory simulation classes
from params import get_params_from_config  # pyright: ignore[reportMissingImports]
from simulate_memory import StageRematMemSimulator  # pyright: ignore[reportMissingImports]

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
	dist.init_process_group(
		backend="nccl",
		device_id=torch.device(f"cuda:{local_rank}"),
		timeout=datetime.timedelta(seconds=300),
	)
	return dist.get_rank(), dist.get_world_size()


def run_memory_benchmark(
	model: FullTransformer,
	parts: Any,
	scheduler: str,
	placement: List[int],
	dtype: torch.dtype,
	rank: int = 0,
) -> Dict[str, Any]:
	"""Run benchmark with detailed memory tracking and return execution data."""
	# Run benchmark to get detailed stats including memory evolution
	all_detailed_stats = bench_with_detailed_memory(model, parts, scheduler, placement, dtype)

	if rank == 0:
		# Structure the data for easy parsing
		measured_data = {
			"detailed_stats_per_rank": all_detailed_stats,
			"num_processors": len(all_detailed_stats),
		}

		return measured_data
	else:
		return {}


def bench_with_detailed_memory(model, parts, scheduler, placement, dtype):
	"""Modified bench function that returns detailed memory statistics."""

	world_size = dist.get_world_size()
	rank = dist.get_rank()

	# Use same parameters as benchmark_utils.bench
	microbatch_size = 1
	n_micro_batches = world_size * 2
	batch_size = microbatch_size * n_micro_batches

	loss_fn = model.loss_fn

	# Create pipeline exactly like benchmark_utils.bench
	sources, dsts = get_sources_targets_sequential(placement)
	pipe = Pipeline(
		parts,
		None,
		placement=placement,
		partitioner=False,
		scheduler=scheduler,
		sources=sources,
		targets=dsts,
	)

	# Warmup iteration
	_ = pipe(
		model.get_sample(batch_size, dtype, device="cuda"),
		model.get_target(batch_size, dtype, device="cuda"),
		loss_fn,
		split_size=microbatch_size,
	)

	torch.cuda.synchronize()
	dist.barrier()
	torch.cuda.reset_peak_memory_stats()

	# Run one iteration with detailed memory tracking

	pipe.zero_grad(set_to_none=False)
	_ = pipe(
		model.get_sample(batch_size, dtype, device="cuda"),
		model.get_target(batch_size, dtype, device="cuda"),
		loss_fn,
		split_size=microbatch_size,
	)

	torch.cuda.synchronize()
	dist.barrier()

	# Access detailed stats from the pipeline
	detailed_stats = getattr(pipe, "detailed_stats", {})

	# Gather all detailed stats to rank 0
	all_detailed_stats = [None] * world_size if rank == 0 else None
	dist.gather_object(detailed_stats, all_detailed_stats, dst=0)

	pipe.clear()

	if rank == 0:
		return all_detailed_stats
	else:
		return None, None, None


def extract_ilp_memory_evolution(solution: Dict, config: str or os.PathLike) -> Dict[str, Any]:
	"""Extract memory evolution from ILP solution using the simulator."""
	# Create params from profiling data
	params = get_params_from_config(config)

	# Get the appropriate simulator
	simulator = StageRematMemSimulator(params, solution)

	# Calculate memory evolution for each processor
	ilp_data = {
		"num_processors": params["p"],
		"memory_evolution_mb": [],  # List of lists, one per processor
		"peak_memory_mb": [],  # List of peak memory per processor
	}

	for i in range(params["p"]):
		evolution, peak = simulator.mem_evolution(i)
		ilp_data["memory_evolution_mb"].append(evolution)
		ilp_data["peak_memory_mb"].append(peak)

	return ilp_data


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--log", choices=["info", "debug", "none"], default="none")
	parser.add_argument(
		"--solution-file", type=str, required=True, help="Path to the ILP solution file"
	)
	parser.add_argument(
		"--output-file", type=str, required=True, help="Path to write comparison results"
	)
	parser.add_argument("--solution-type", type=str, required=True, help="Solution type to analyze")
	add_transformer_args(parser)
	args = parser.parse_args()
	setup_logging(args.log)

	rank, world_size = setup_distributed()

	# Ensure output directory exists
	if rank == 0:
		os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

	# Load solution
	with open(args.solution_file, "r") as f:
		solution_data = json.load(f)
		solution = solution_data.get("solutions", {}).get(args.solution_type)

	if solution is None:
		if rank == 0:
			print("No solution found in solution file")
		sys.exit(1)

	# Create and initialize model
	config = model_config_from_args(args)
	dtype = config.pop("dtype")
	with torch.device("meta"):
		model = FullTransformer(**config).to(dtype)
		replace_linear_with_linear_dw(model, "meta")

	if rank == 0:
		print(f"\nModel has {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

	try:
		# Extract ILP memory estimates
		if rank == 0:
			print("Extracting ILP memory estimates...")
		ilp_memory_data = extract_ilp_memory_evolution(solution, args.config_file)

		# Run actual benchmark with memory tracking
		if rank == 0:
			print("Running actual benchmark with memory tracking...")

		placement = solution.get("placement")
		balance = balanced_partition(config["n_blocks"], placement)
		scheduler = RematScheduler(solution)
		parts = get_handcrafted_imbalanced_partition(model, rank, placement, balance)

		measured_data = run_memory_benchmark(model, parts, scheduler, placement, dtype, rank=rank)

		# Prepare final results (only on rank 0)
		if rank == 0:
			results = {"ilp_estimates": ilp_memory_data, "measured_data": measured_data}

			# Save results
			with open(args.output_file, "w") as f:
				json.dump(results, f, indent=4)

			print(f"\nResults saved to: {args.output_file}")

	except torch.cuda.OutOfMemoryError:
		print(f"Out of memory on rank {rank}")
		sys.exit(1)

	# Print full stacktrace for any exceptions
	except Exception as e:
		raise e

	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
