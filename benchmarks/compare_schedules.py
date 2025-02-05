import torch
import torch.distributed as dist
import os
import sys
import gc


sys.path.append("./")
from elf.pipeline import Pipeline, get_sources_targets_sequential
from argparse import ArgumentParser
from models.simple import FullTransformer
from elf.utils import Timer, pretty_print_params
from elf.zb_utils import replace_linear_with_linear_dw

import logging

logger = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO)


def get_handcrafted_partition(model, rank, placement):
	# CRAFTED FOR FULLTRANSFORMER, ADAPT FOR OTHER MODELS
	num_blocks = len(model.blocks)
	num_ranks = len(placement)
	blocks_per_rank = num_blocks // num_ranks
	extra_blocks = num_blocks % num_ranks
	parts = [None] * num_ranks

	start_idx = 0
	for i in range(num_ranks):
		# Add one extra block to earlier ranks if blocks don't divide evenly
		rank_blocks = blocks_per_rank + (1 if i < extra_blocks else 0)
		end_idx = start_idx + rank_blocks

		if i == 0:
			parts[i] = torch.nn.Sequential(model.embed, *model.blocks[start_idx:end_idx])
		elif i == num_ranks - 1:
			parts[i] = torch.nn.Sequential(*model.blocks[start_idx:], model.head)
		else:
			parts[i] = torch.nn.Sequential(*model.blocks[start_idx:end_idx])

		start_idx = end_idx

	return [parts[i] for i, p in enumerate(placement) if p == rank]


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
	rank = int(os.environ["LOCAL_RANK"])
	global_rank = int(os.environ["RANK"])

	torch.cuda.set_device(rank)
	fileout = "results.csv"

	dist.init_process_group(backend="nccl")

	if global_rank == 0:
		if os.path.exists(fileout):
			os.remove(fileout)
		with open(fileout, "w") as f:
			f.write("name")
			for i in range(4):
				f.write(f",total_time_{i},idle_time_{i},start_time_{i},end_time_{i},bubble_time_{i}")
			for i in range(4):
				f.write(f",mem_{i}")
			f.write("\n")

	# torch.cuda.cudart().cudaProfilerStart()

	model = FullTransformer(500, 1024, 32, 256, 32, 0.1)
	if rank == 0:
		print(f"Model has {pretty_print_params(sum(p.numel() for p in model.parameters()))} parameters")
	loss_fn = model.loss_fn

	setups = [
		("GPipe", list(range(world_size)), "afab"),
		("1f1b", list(range(world_size)), "1f1b"),
		# ("Megatron", list(range(world_size)) * 2, "1f1b"),
		("Hanayo", list(range(world_size)) + list(reversed(range(world_size))), "hanayo"),
		("Full Remat", list(range(world_size)), "full_remat"),
		("ZBH1", list(range(world_size)), "zbh1"),
		("ZBH2", list(range(world_size)), "zbh2"),
	]

	batch_size = 32
	n_micro_batches = 8
	split_size = batch_size // n_micro_batches
	n_iterations = 20

	for s, placement, schedule in setups:
		partitioner = args.partitioner
		if partitioner == "handcrafted":
			parts = get_handcrafted_partition(model, rank, placement)
			sources, dsts = get_sources_targets_sequential(placement)  # "targets" is already used :)
			partitioner = False
		else:
			parts = model
			sources, dsts = None, None

		if s.startswith("ZBH"):
			replace_linear_with_linear_dw(model, rank)

		if global_rank == 0:
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
		# Warmup
		if global_rank == 0:
			print(f"{s} - Warming up")
		for i in range(3):
			y = pipe(inputs.clone(), targets.clone(), loss_fn, split_size=split_size)

		torch.cuda.reset_peak_memory_stats()

		if global_rank == 0:
			print(f"{s} - Benchmark")

		dist.barrier()
		torch.cuda.synchronize()

		stats = []
		with Timer() as timer:
			for i in range(n_iterations):
				model.zero_grad()
				inputs = model.get_sample(batch_size)  # should we include input allocation in the stats?
				targets = model.get_target(batch_size)
				y = pipe(inputs, targets, loss_fn, split_size=split_size)
				stats.append(pipe.stats)
			dist.barrier()

		mems = [torch.tensor(0.0, device=rank) for _ in range(world_size)] if global_rank == 0 else None
		dist.gather(torch.tensor(torch.cuda.max_memory_allocated() / (2**30), device=rank), mems, 0)

		median_times = medians(stats)
		itimes = [{} for _ in range(world_size)] if global_rank == 0 else None
		dist.gather_object(median_times, itimes, 0)

		if global_rank == 0:
			iteration_times = [f"{it['total']:.2f}" for it in itimes]
			idle_times = [f"{it['idle']:.2f}" for it in itimes]
			idle_percentages = [f"{it['idle'] / it['total'] * 100:.1f}" for it in itimes]
			peak_mems = [f"{m.item():.2f}" for m in mems]
			print(f"{s}:")
			print(f"\tIteration times: {iteration_times} s")
			print(f"\tIdle times: {idle_times} s ({idle_percentages}%)")
			print(f"\tTotal time ({n_iterations} iterations): {timer.time():.2f} s")
			print(f"\tPeak memories: {peak_mems} GB")

			with open(fileout, "a") as f:
				f.write(f"{s}")
				for d in itimes:
					for t in d.values():
						f.write(f",{t}")
				for m in mems:
					f.write(f",{m}")
				f.write("\n")
				f.flush()

		pipe.clear()
		model.zero_grad(set_to_none=True)
		del pipe, y, inputs, targets, median_times, mems, itimes
		gc.collect()
		torch.cuda.empty_cache()

	# torch.cuda.cudart().cudaProfilerStop()

	dist.barrier()
	if dist.is_initialized():
		dist.destroy_process_group()
