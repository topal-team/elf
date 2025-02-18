import os
import gc
import sys
from datetime import timedelta
from argparse import ArgumentParser

import torch
import torch.distributed as dist

sys.path.append(".")
from elf.pipeline import Pipeline, get_sources_targets_sequential
from elf.scheduling import OpOptions, OperationType
from models.simple import ChainTransformer, Attention
from elf.utils import Timer, pretty_print_params
from elf.zb_utils import replace_linear_with_linear_dw
from elf.schedules import generate_zbh2_schedule, schedule_from_str

import logging

logger = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO)


def get_handcrafted_partition(model, rank, placement):
	# CRAFTED FOR CHAINTRANSFORMER, ADAPT FOR OTHER MODELS
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

		parts[i] = torch.nn.Sequential(*model.blocks[start_idx:end_idx])

		start_idx = end_idx

	return [parts[i] for i, p in enumerate(placement) if p == rank]

def get_handcrafted_imbalanced_partition(model, rank, placement, factors):

	num_blocks = len(model.blocks)
	num_ranks = len(placement)
	parts = [None] * num_ranks
	assert int(sum(factors)) == int(num_blocks)

	start_idx = 0
	for i in range(num_ranks):
		end_idx = start_idx + factors[i]
		parts[i] = torch.nn.Sequential(*model.blocks[start_idx:end_idx])
		start_idx = end_idx

	return [parts[i] for i, p in enumerate(placement) if p == rank]
	

def manual_zb(placement, nmb, signatures):
	# sched0 = "ffffFFFbwfbwbwbwrbwrbwrbwbw"
	# sched1 = "ffffFbFbfbwfbwrbwrbwbwbwww"
	# sched2 = "fffbfbfbfbfbwfbwbwbwwwww"
	# sched3 = "fbfbfbfbfbfbfbwfbwwwwwww"
	sched0 = "fFFFFFFrbwFbwrbwrrbwrbwrbwrbwbw"
	sched1 = "fFFFFrbFrbFwbwFrbrbwrbwrbwrbwww"
	sched2 = "fFFbFrbFrbFrbFrbFwrbwrbwrbwwwww"
	sched3 = "fbfbfbfbfbfbwfbwfbwwwwww"
	s = [sched0, sched1, sched2, sched3]

	sched = schedule_from_str(s, placement, signatures)
	return sched

def sr_zb(placement, nmb, signatures):
	mbs = [
		[0, 1, 5, 6],
		[6],
		[],
		[]
	]

	sched = generate_zbh2_schedule(placement, nmb, signatures)
	for op in sched:
		if op.mb_id in mbs[op.block_id] and op.op in [OperationType.FORWARD, OperationType.BACKWARD_INPUTS]:
			op.options[OpOptions.REMAT_STRATEGY] = "selective"
			op.options[OpOptions.REMAT_SELECTION] = Attention

	return sched


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
	fileout = "results.csv"

	dist.init_process_group(backend="nccl", timeout=timedelta(seconds=300))

	if rank == 0:
		if os.path.exists(fileout):
			os.remove(fileout)
		with open(fileout, "w") as f:
			f.write("name")
			for i in range(world_size):
				f.write(f",total_time_{i},idle_time_{i},start_time_{i},end_time_{i},bubble_time_{i}")
			for i in range(world_size):
				f.write(f",mem_{i}")
			f.write("\n")

	# torch.cuda.cudart().cudaProfilerStart()

	n_blocks = 224

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
		# ("ZBH1", list(range(world_size)), "zbh1"),
		# ("SRILP-ZBH2", list(range(world_size)), sr_zb),
		# ("ZBH2", list(range(world_size)), "zbh2"),
		("Imb-ZBH2", list(range(world_size)), "zbh2"),
		# ("Manual ZBH2", list(range(world_size)), manual_zb),
	]

	split_size = 2
	n_micro_batches = world_size * 2
	batch_size = split_size * n_micro_batches
	n_iterations = 20

	replaced_dw = False

	for s, placement, schedule in setups:
		model.cpu()
		partitioner = args.partitioner
		if partitioner == "handcrafted":
			if "Imb" in s:
				factors = [53, 58, 57, 56]
				parts = get_handcrafted_imbalanced_partition(model, rank, placement, factors)
			else:
				parts = get_handcrafted_partition(model, rank, placement)
			sources, dsts = get_sources_targets_sequential(placement)  # "targets" is already used :)
			partitioner = False
		else:
			parts = model
			sources, dsts = None, None

		if "ZBH" in s and not replaced_dw:
			replaced_dw = True
			replace_linear_with_linear_dw(model, 'cpu')

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
			available_mem = torch.cuda.get_device_properties(local_rank).total_memory - torch.cuda.memory_allocated()
			print(f"Available memory: {available_mem / (2**30):.2f}GB")
			print(f"Allocated memory: {torch.cuda.memory_allocated() / (2**30):.2f}GB")
			print(f"Occupied by parameters: {sum(p.numel() * p.element_size() for p in pipe.blocks[0].model.parameters()) / (2**30):.2f}GB")
		
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
			available_mem = torch.cuda.get_device_properties(local_rank).total_memory - torch.cuda.memory_allocated()
			print(f"Available memory: {available_mem / (2**30):.2f}GB")
			print(f"Allocated memory: {torch.cuda.memory_allocated() / (2**30):.2f}GB")
			print(f"Occupied by parameters: {sum((p.numel() + p.grad.numel()) * p.element_size() for p in pipe.blocks[0].model.parameters()) / (2**30):.2f}GB")

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
		dist.gather(torch.tensor(torch.cuda.max_memory_allocated() / (2**30), device=local_rank), mems, 0)

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
			print(f"\tTotal time ({n_iterations} iterations): {timer.time():.2f}s - Throughput: {(n_iterations * batch_size / timer.time()):.2f} seq/s, Time / iter: {timer.time() / n_iterations:.2f}s")
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
