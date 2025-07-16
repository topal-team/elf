import json
import os
import gc
import sys
import logging

from datetime import timedelta
from argparse import ArgumentParser

import numpy as np
import torch
import torch.distributed as dist

sys.path.append(".")
from elf import Pipeline, get_sources_targets_sequential, replace_linear_with_linear_dw
from elf.utils import Timer, pretty_print_params
from models.utils import add_transformer_args, build_model_from_args
from benchmarks.benchmark_utils import get_handcrafted_imbalanced_partition, meta_to_device

logger = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
	parser = ArgumentParser(description="Benchmark different schedules")
	parser.add_argument(
		"--log", choices=["debug", "info", "none"], default="none", required=False, help="logging level"
	)
	parser.add_argument(
		"--partitioner",
		choices=["naive", "constrained", "metis", "dagP", "handcrafted"],
		required=False,
		default="handcrafted",
		help="partitioner to distribute the model",
	)
	add_transformer_args(parser, model_type="chain")
	parser.set_defaults(hidden_dim=2048, nblocks=16, seq_len=512, num_heads=32, dropout=0.1)

	parser.add_argument("--niters", type=int, default=30, required=False, help="number of iterations")
	parser.add_argument(
		"--output",
		type=str,
		default="results/compare_schedules.json",
		required=False,
		help="output file",
	)
	parser.add_argument(
		"--ntrials", type=int, default=10, required=False, help="number of trials to run for error bars"
	)
	args = parser.parse_args()
	match args.log:
		case "debug":
			logging.getLogger().setLevel(logging.DEBUG)
		case "info":
			logging.getLogger().setLevel(logging.INFO)
		case "none":
			logging.getLogger().setLevel(100)

	world_size = int(os.environ["WORLD_SIZE"])
	rank = int(os.environ["RANK"])
	local_rank = int(os.environ["LOCAL_RANK"])

	torch.cuda.set_device(local_rank)

	dist.init_process_group(
		backend="nccl", device_id=torch.device(local_rank), timeout=timedelta(seconds=300)
	)

	# torch.cuda.cudart().cudaProfilerStart()

	with torch.device("meta"):
		model, dtype = build_model_from_args(args, model_type="chain")

	if rank == 0:
		print(f"Model has {pretty_print_params(sum(p.numel() for p in model.parameters()))} parameters")

	loss_fn = model.loss_fn

	setups = [
		("GPipe", list(range(world_size)), "afab"),
		("1f1b", list(range(world_size)), "1f1b"),
		("Megatron", list(range(world_size)) * 2, "1f1b"),
		("Megatron x2", list(range(world_size)) * 4, "1f1b"),
		("Hanayo 1W", list(range(world_size)) + list(reversed(range(world_size))), "hanayo"),
		("Hanayo 2W", (list(range(world_size)) + list(reversed(range(world_size)))) * 2, "hanayo"),
		("Hanayo 4W", (list(range(world_size)) + list(reversed(range(world_size)))) * 4, "hanayo"),
		# ("Full Remat", list(range(world_size)), "full_remat"),
		("ZBH1", list(range(world_size)), "zbh1"),
		("ZBH2", list(range(world_size)), "zbh2"),
		("ZBV", list(range(world_size)) + list(reversed(range(world_size))), "zbv"),
	]
	n_micro_batches = world_size * 2
	split_size = 1
	batch_size = split_size * n_micro_batches
	n_iterations = args.niters

	replaced_dw = False

	results = {}

	for s, placement, scheduler in setups:
		nparts = len(placement)
		partitioner = args.partitioner
		if partitioner == "handcrafted":
			factors = [
				(args.nblocks // nparts) + (1 if i < args.nblocks % nparts else 0) for i in range(nparts)
			]
			parts = get_handcrafted_imbalanced_partition(model, rank, placement, factors)
			sources, dsts = get_sources_targets_sequential(placement)  # "targets" is already used :)
			partitioner = False
		else:
			parts = model
			sources, dsts = None, None
			if rank == 0:
				model = meta_to_device(model)

		if "ZB" in s and not replaced_dw:
			replaced_dw = True
			replace_linear_with_linear_dw(model, "meta")

		if rank == 0:
			print(f"Beginning benchmark for {s}")

		inputs = model.get_sample(batch_size, dtype)
		targets = model.get_target(batch_size, dtype)

		pipe = Pipeline(
			parts,
			inputs,
			placement,
			scheduler=scheduler,
			partitioner=partitioner,
			sources=sources,
			targets=dsts,
		)

		# Warmup
		if rank == 0:
			print(f"{s} - Warming up")

		for i in range(3):
			y, loss = pipe(inputs.clone(), targets.clone(), loss_fn, split_size=split_size)
			del y, loss

		torch.cuda.reset_peak_memory_stats()

		if rank == 0:
			print(f"{s} - Benchmark")

		dist.barrier()
		torch.cuda.synchronize()

		trial_times = []
		for trial in range(args.ntrials):
			stats = []
			with Timer() as timer:
				for i in range(n_iterations):
					model.zero_grad()
					inputs = model.get_sample(batch_size, dtype)
					targets = model.get_target(batch_size, dtype)
					_ = pipe(inputs, targets, loss_fn, split_size=split_size)
					stats.append(pipe.stats)

				dist.barrier()

			trial_times.append(timer.time())

		if rank == 0:
			mean_time = np.mean(trial_times)
			std_time = np.std(trial_times)
			mean_throughput = n_iterations * batch_size / mean_time
			std_throughput = mean_throughput * (std_time / mean_time)

			print(f"{s}:")
			print(
				f"\tTotal time ({n_iterations} iterations, {args.ntrials} trials): {mean_time:.2f} ± {std_time:.2f}s"
			)
			print(f"\tThroughput: {mean_throughput:.2f} ± {std_throughput:.2f} seq/s")
			print(f"\tTime per iter: {mean_time / n_iterations:.2f} ± {std_time / n_iterations:.2f}s")

			results[s] = {
				"placement": placement,
				"scheduler": scheduler,
				"total_time_mean": mean_time,
				"total_time_std": std_time,
				"throughput_mean": mean_throughput,
				"throughput_std": std_throughput,
				"time_per_iter_mean": mean_time / n_iterations,
				"time_per_iter_std": std_time / n_iterations,
			}

		pipe.clear()
		model.zero_grad(set_to_none=True)
		del pipe, inputs, targets
		gc.collect()
		torch.cuda.empty_cache()

	# torch.cuda.cudart().cudaProfilerStop()

	if rank == 0:
		with open(args.output, "w") as f:
			json.dump(results, f)

	dist.barrier()
	if dist.is_initialized():
		dist.destroy_process_group()
