import os
import gc
import sys
from datetime import timedelta
from argparse import ArgumentParser

import torch
import torch.distributed as dist

sys.path.append(".")
from elf.pipeline import Pipeline, get_sources_targets_sequential
from models.simple import ChainTransformer
from elf.utils import Timer, pretty_print_params
from elf.zb_utils import replace_linear_with_linear_dw
from benchmarks.benchmark_utils import get_handcrafted_imbalanced_partition

import logging

logger = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO)


def medians(times):
	meds = {}
	for t in times[0].keys():
		values = list(map(lambda x: x[t], times))
		meds[t] = sorted(values)[len(values) // 2]
	return meds


if __name__ == "__main__":
	parser = ArgumentParser(description="Benchmark different schedules")
	parser.add_argument(
		"--log", choices=["debug", "info", "none"], default="info", required=False, help="logging level"
	)
	parser.add_argument(
		"--partitioner",
		choices=["naive", "constrained", "metis", "dagP", "handcrafted"],
		required=False,
		default="naive",
		help="partitioner to distribute the model",
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

	dist.init_process_group(backend="nccl", timeout=timedelta(seconds=300))

	# torch.cuda.cudart().cudaProfilerStart()

	n_blocks = 128

	model = ChainTransformer(1024, n_blocks, 256, 32, 0.1)
	if rank == 0:
		print(f"Model has {pretty_print_params(sum(p.numel() for p in model.parameters()))} parameters")
	loss_fn = model.loss_fn

	setups = [
		# ("GPipe", list(range(world_size)), "afab"),
		# ("1f1b", list(range(world_size)), "1f1b"),
		# ("Megatron", list(range(world_size)) * 2, "1f1b"),
		# ("Hanayo 1W", list(range(world_size)) + list(reversed(range(world_size))), "hanayo"),
		# ("Hanayo 2W", (list(range(world_size)) + list(reversed(range(world_size)))) * 2, "hanayo"),
		# ("Full Remat", list(range(world_size)), "full_remat"),
		("ZBH1", list(range(world_size)), "zbh1"),
		("ZBH2", list(range(world_size)), "zbh2"),
		("ZBV", list(range(world_size)) + list(reversed(range(world_size))), "zbv"),
	]
	n_micro_batches = world_size * 2
	split_size = 2
	batch_size = split_size * n_micro_batches
	n_iterations = 20

	replaced_dw = False

	for s, placement, schedule in setups:
		nparts = len(placement)
		model.cpu()
		partitioner = args.partitioner
		if partitioner == "handcrafted":
			factors = [n_blocks // nparts for _ in range(nparts)]
			parts = get_handcrafted_imbalanced_partition(model, rank, placement, factors)
			sources, dsts = get_sources_targets_sequential(placement)  # "targets" is already used :)
			partitioner = False
		else:
			parts = model
			sources, dsts = None, None

		if "ZB" in s and not replaced_dw:
			replaced_dw = True
			replace_linear_with_linear_dw(model, "cpu")

		if rank == 0:
			print(f"Beginning benchmark for {s}")

		inputs = model.get_sample(batch_size)
		targets = model.get_target(batch_size)

		pipe = Pipeline(
			parts,
			inputs,
			placement,
			schedule=schedule,
			partitioner=partitioner,
			sources=sources,
			targets=dsts,
		)

		if rank == 0:
			available_mem = (
				torch.cuda.get_device_properties(local_rank).total_memory - torch.cuda.memory_allocated()
			)
			print(f"Available memory: {available_mem / (2**30):.2f}GB")
			print(f"Allocated memory: {torch.cuda.memory_allocated() / (2**30):.2f}GB")
			print(
				f"Occupied by parameters: {sum(p.numel() * p.element_size() for p in pipe.blocks[0].model.parameters()) / (2**30):.2f}GB"
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

		if rank == 0:
			available_mem = (
				torch.cuda.get_device_properties(local_rank).total_memory - torch.cuda.memory_allocated()
			)
			print(f"Available memory: {available_mem / (2**30):.2f}GB")
			print(f"Allocated memory: {torch.cuda.memory_allocated() / (2**30):.2f}GB")
			print(
				f"Occupied by parameters: {sum((p.numel() + p.grad.numel()) * p.element_size() for p in pipe.blocks[0].model.parameters()) / (2**30):.2f}GB"
			)

		stats = []
		with Timer() as timer:
			for i in range(n_iterations):
				model.zero_grad()
				inputs = model.get_sample(batch_size)  # should we include input allocation in the stats?
				targets = model.get_target(batch_size)
				y = pipe(inputs, targets, loss_fn, split_size=split_size)
				stats.append(pipe.stats)
			dist.barrier()

		mems = [torch.tensor(0.0, device=local_rank) for _ in range(world_size)] if rank == 0 else None
		dist.gather(
			torch.tensor(torch.cuda.max_memory_allocated() / (2**30), device=local_rank), mems, 0
		)

		median_times = medians(stats)
		itimes = [{} for _ in range(world_size)] if rank == 0 else None
		dist.gather_object(median_times, itimes, 0)

		if rank == 0:
			iteration_times = [f"{it['total']:.2f}" for it in itimes]
			idle_times = [f"{it['idle']:.2f}" for it in itimes]
			idle_percentages = [f"{it['idle'] / it['total'] * 100:.1f}" for it in itimes]
			peak_mems = [f"{m.item():.2f}" for m in mems]
			print(f"{s}:")
			print(f"\tIteration times: {iteration_times} s")
			print(f"\tIdle times: {idle_times} s ({idle_percentages}%)")
			print(
				f"\tTotal time ({n_iterations} iterations): {timer.time():.2f}s - Throughput: {(n_iterations * batch_size / timer.time()):.2f} seq/s, Time / iter: {timer.time() / n_iterations:.2f}s"
			)
			print(f"\tPeak memories: {peak_mems} GB")

		pipe.clear()
		model.zero_grad(set_to_none=True)
		del pipe, y, inputs, targets, median_times, mems, itimes
		gc.collect()
		torch.cuda.empty_cache()

	# torch.cuda.cudart().cudaProfilerStop()

	dist.barrier()
	if dist.is_initialized():
		dist.destroy_process_group()
