import torch
import torch.distributed as dist
import os
import sys
import gc

sys.path.append("./")
from pipeline.pipeline import Pipeline
from argparse import ArgumentParser
from models.simple import SimpleTransformer
from pipeline.utils import pretty_print_params

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

	model = SimpleTransformer(8000, 2048, 32, 512)
	if rank == 0:
		logger.info(f"Model has {pretty_print_params(sum(p.numel() for p in model.parameters()))} parameters")
	loss_fn = model.loss_fn

	setups = [
		("GPipe", list(range(world_size)), "afab"),
		("1f1b", list(range(world_size)), "1f1b"),
		("Megatron", list(range(world_size)) * 2, "1f1b"),
		("Hanayo", list(range(world_size)) + list(reversed(range(world_size))), "hanayo"),
		("Full Remat", list(range(world_size)), "full_remat"),
	]

	for s, placement, schedule in setups:
		if global_rank == 0:
			logger.info(f"Beginning benchmark for {s}")
		# logger.info(f"Rank {rank} - Memory allocated : {torch.cuda.memory_allocated() / 2**30:.3f} GB")

		inputs = model.get_sample(32)
		targets = model.get_target(32)

		pipe = Pipeline(model, inputs, placement, schedule=schedule)
		# Warmup
		if global_rank == 0:
			logger.info(f"{s} - Warming up")
		for i in range(3):
			y = pipe(inputs.clone(), targets.clone(), loss_fn)
		torch.cuda.reset_peak_memory_stats()

		if global_rank == 0:
			logger.info(f"{s} - Benchmark")
		times = []
		for i in range(10):
			model.zero_grad()
			inputs = model.get_sample(32)
			targets = model.get_target(32)
			y = pipe(inputs, targets, loss_fn)
			times.append(pipe.times)

		mems = (
			[torch.tensor(0.0, device=rank) for _ in range(world_size)] if global_rank == 0 else None
		)
		dist.gather(torch.tensor(torch.cuda.max_memory_allocated() / (2**30), device=rank), mems, 0)

		median_times = medians(times)
		itimes = [{} for _ in range(world_size)] if global_rank == 0 else None
		dist.gather_object(median_times, itimes, 0)

		if global_rank == 0:
			print(
				f'{s} :\n\tMedian time : {median_times["total"]:.3f}s\n\tIdle time : {median_times["idle"]:.2f}s, or {100 * median_times["idle"] / median_times["total"]:.1f}% of total time.\n\tMems : {[m.item() for m in mems]} GB'
			)
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
