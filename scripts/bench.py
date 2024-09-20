import torch
import torch.distributed as dist
import os
import sys

sys.path.append("./")
from pipeline.pipeline import Pipeline
from argparse import ArgumentParser
from settings import *

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
	parser = ArgumentParser(description="Benchmark of pipelined model with custom engine")
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
		with open(fileout, "w") as f:
			f.write(
				"name,mb_size,total_time_0,idle_time_0,start_time_0,end_time_0,bubble_time_0,total_time_1,idle_time_1,start_time_1,end_time_1,bubble_time_1,total_time_2,idle_time_2,start_time_2,end_time_2,bubble_time_2,total_time_3,idle_time_3,start_time_3,end_time_3,bubble_time_3,mem_0,mem_1,mem_2,mem_3\n"
			)

	torch.cuda.cudart().cudaProfilerStart()

	inputs = inputs.cuda()

	for s, placement, schedule in setups:
		pipe = Pipeline(model, inputs, placement, schedule=schedule)
		for size in split_sizes:
			# if global_rank == 0: print(f'Memory allocated : {torch.cuda.memory_allocated() / 2**30:.3f} GB')

			if global_rank == 0:
				logger.info(f"{s} - Beginning bench for micro batches of size {size}")

			# Warmup
			if global_rank == 0:
				logger.info(f"{s} - Warming up")
			for i in range(warmups):
				_ = pipe(inputs.clone(), torch.empty(0), lambda x, _: x.sum(), size, **options)
			torch.cuda.reset_peak_memory_stats()

			if global_rank == 0:
				logger.info(f"{s} - Benchmark")
			times = []
			for i in range(iters):
				_ = pipe(inputs.detach(), torch.empty(0), lambda x, _: x.sum(), size, **options)
				model.zero_grad()
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
					f'{s} - Size {size} :\n\tMedian time : {median_times["total"]:.3f}s\n\tIdle time : {median_times["idle"]:.2f}s, or {100 * median_times["idle"] / median_times["total"]:.1f}% of total time.'
				)
				with open(fileout, "a") as f:
					f.write(f"{s},{size}")
					for d in itimes:
						for t in d.values():
							f.write(f",{t}")
					for m in mems:
						f.write(f",{m}")
					f.write("\n")
					f.flush()

	torch.cuda.cudart().cudaProfilerStop()

	dist.barrier()
	if dist.is_initialized():
		dist.destroy_process_group()
