import os
import gc
import sys
import json
import torch
import torch.distributed as dist
import argparse
import logging
import psutil
from typing import Dict, List, Optional, Any

sys.path.append(".")
from elf.zb_utils import replace_linear_with_linear_dw
from models.simple import ChainTransformer, FullTransformer  # noqa: F401
from benchmarks.benchmark_utils import bench, get_handcrafted_imbalanced_partition
from benchmarks.zb_schedulers import FullRematScheduler, PartialRematScheduler

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
	dist.init_process_group(backend="nccl")
	return dist.get_rank(), dist.get_world_size()


def load_results(restart: bool) -> Dict:
	"""Load existing results or create new results dictionary."""
	if restart:
		return {}
	with open("results/ilps-benchmark.json", "r") as f:
		return json.load(f)


def create_model(
	hidden_size: int, n: int, seq_len: int, num_heads: int, dropout: float
) -> ChainTransformer:
	"""Create and initialize the model."""
	model = ChainTransformer(hidden_size, n, seq_len, num_heads, dropout)
	replace_linear_with_linear_dw(model, "cpu")
	for param in model.parameters():
		tensor = param.data.cuda()
		dist.broadcast(tensor, src=0)
		param.data = tensor.cpu()
	return model


def log_model_info(model: ChainTransformer, n: int, rank: int) -> None:
	"""Log model information."""
	if rank == 0:
		print(
			f"\nWith {n} blocks, model has {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters"
		)
		allocated = torch.cuda.memory_allocated() / (1024**3)
		free = torch.cuda.get_device_properties().total_memory / (1024**3) - allocated
		print(f"Memory allocated: {allocated:.2f}GB")
		print(f"Memory free: {free:.2f}GB")


def run_benchmark(
	model: ChainTransformer, parts: Any, scheduler: str, grad_acc: int = 1, rank: int = 0
) -> tuple[float, List[float]]:
	"""Run benchmark and return iteration time and peak memory usage."""
	iter_time, all_peak_mems = bench(model, parts, scheduler, grad_acc)
	if rank == 0:
		print(f"\t{iter_time:.2f}s, Peak memory: {[f'{m:.2f}' for m in all_peak_mems]} GB")
	return iter_time, all_peak_mems


def process_solution(
	model: ChainTransformer,
	solution: Optional[Dict],
	solution_type: str,
	base: str,
	rank: int,
	world_size: int,
	n: int,
) -> tuple[Optional[float], Optional[List[float]]]:
	"""Process a single solution type and return benchmark results."""
	if solution is None:
		if rank == 0:
			print(f"No {solution_type} solution found for n = {n}")
		return None, None

	if rank == 0:
		print(f"{solution_type.capitalize()}:")

	if solution_type in ["remat", "rematf"]:
		balance = balanced_partition(n, world_size)
		scheduler = FullRematScheduler(solution, base, balance)
	elif solution_type in ["combined", "combinedf"]:
		balance = [int(b) for b in solution["b"]]
		scheduler = PartialRematScheduler(solution, base)
	elif solution_type in ["balance"]:
		balance = [int(b) for b in solution["b"]]
		scheduler = base

	parts = get_handcrafted_imbalanced_partition(model, rank, list(range(world_size)), balance)
	return run_benchmark(model, parts, scheduler, rank=rank)

def balanced_partition(n: int, world_size: int) -> List[int]:
	"""Distribute blocks evenly, handling remainder."""
	blocks_per_gpu = n // world_size
	remainder = n % world_size
	return [blocks_per_gpu + (1 if i < remainder else 0) for i in range(world_size)]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--restart", action="store_true", default=False)
	parser.add_argument("--log", choices=["info", "debug", "none"], default="none")
	args = parser.parse_args()
	setup_logging(args.log)

	rank, world_size = setup_distributed()

	# Model configuration
	hidden_size = 2304
	seq_len = 1024
	num_heads = 24
	dropout = 0.1
	base = "zbh2"

	# Load solutions and results
	with open("local/ilp-solutions.json", "r") as f:
		ilp_solutions = json.load(f)
	results = load_results(args.restart)

	types = [
		"base",  # equal balancing, no recomputation
		"remat",  # equal balancing, all recomputations enabled
		"rematf",  # equal balancing, only forward recomputations enabled
		"balance",  # load balancing, no recomputation
		"combined",  # load balancing, all recomputations enabled
		"combinedf",  # load balancing, only forward recomputations enabled
	]

	for n in ilp_solutions:
		if n in results:
			continue

		n = int(n)
		results[str(n)] = {key: {"time": 0.0, "peak_mems": [0.0]} for key in types}
		solutions = ilp_solutions[str(n)]

		# Create and initialize model
		model = create_model(hidden_size, n, seq_len, num_heads, dropout)
		log_model_info(model, n, rank)

		# Check if base is possible
		base_is_possible = (
			all(all(sum(r) == 0 for r in remat_type) for remat_type in solutions["remat"].values())
			if "remat" in solutions
			else False
		)

		# Run base benchmark
		if base_is_possible:  # 256 for zbh2
			# Distribute blocks evenly, handling remainder
			balance = balanced_partition(n, world_size)
			parts = get_handcrafted_imbalanced_partition(model, rank, list(range(world_size)), balance)
			iter_time, all_peak_mems = run_benchmark(model, parts, base, 1, rank)
			if rank == 0:
				results[str(n)]["base"]["time"] = iter_time
				results[str(n)]["base"]["peak_mems"] = [float(m) for m in all_peak_mems]

			del parts

		else:
			if rank == 0:
				print(f"Base is not possible for n = {n}")

		# Process other solution types
		for solution_type in ["balance", "remat", "rematf", "combined", "combinedf"]:
			model.zero_grad()
			model.cpu()

			if rank == 0:
				gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
				process = psutil.Process(os.getpid())
				cpu_allocated = process.memory_info().rss / (1024**3)
				print(
					f"Memory allocated before {solution_type}: GPU {gpu_allocated:.2f}GB, CPU {cpu_allocated:.2f}GB"
				)

			solution = solutions.get(solution_type)
			iter_time, peak_mems = process_solution(
				model, solution, solution_type, base, rank, world_size, n
			)

			if rank == 0 and iter_time is not None:
				results[str(n)][solution_type]["time"] = iter_time
				results[str(n)][solution_type]["peak_mems"] = [float(m) for m in peak_mems]

		# Save results
		if rank == 0:
			with open("results/ilps-benchmark.json", "w") as f:
				json.dump(results, f, indent=4)

		# Cleanup
		del model
		torch.cuda.empty_cache()
		gc.collect()
		dist.barrier()

	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
