import sys
import subprocess

sys.path.append(".")
from pipeline.pipeline import Pipeline


import os
import torch
import torch.distributed as dist
import logging
from argparse import ArgumentParser
import gc
import wandb

from models.GPT import GPTTinyConfig, GPTSmallConfig, GPTMediumConfig, GPTLargeConfig, GPT
from pipeline.utils import pretty_print_params

logger = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO)


def medians(times):
	meds = {}
	for t in times[0].keys():
		values = list(map(lambda x: x[t], times))
		meds[t] = sorted(values)[len(values) // 2]
	return meds


if __name__ == "__main__":
	parser = ArgumentParser(description="Benchmark GPT training")
	parser.add_argument(
		"--model", choices=["tiny", "small", "medium", "large"], default="tiny", help="Model size to benchmark"
	)
	parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
	parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
	parser.add_argument(
		"--log", choices=["debug", "info", "none"], default="info", help="Logging level"
	)
	parser.add_argument(
		"--schedule", choices=["afab", "1f1b", "hanayo", "full_remat"], default="1f1b", help="Schedule to use"
	)
	parser.add_argument("--partitioner", choices=["naive", "constrained", "metis", "dagP"], default="metis", help="Partitioner to use")
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
	dist.init_process_group(backend="nccl")

	# Select model config
	match args.model:
		case "tiny":
			config = GPTTinyConfig(vocab_size=50257, block_size=args.seq_len)
		case "small":
			config = GPTSmallConfig(vocab_size=50257, block_size=args.seq_len)
		case "medium":
			config = GPTMediumConfig(vocab_size=50257, block_size=args.seq_len)
		case "large":
			config = GPTLargeConfig(vocab_size=50257, block_size=args.seq_len)

	# Create model
	model = GPT(config)
	if rank == 0:
		logger.info(
			f"Model has {pretty_print_params(sum(p.numel() for p in model.parameters()))} parameters"
		)

	# Create sample inputs
	inputs = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=rank)
	targets = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=rank)

	# Create pipeline
	pipe = Pipeline(model, inputs, schedule=args.schedule, partitioner=args.partitioner)

	# Warmup
	if global_rank == 0:
		logger.info("Warming up")
	for i in range(3):
		y = pipe(inputs.clone(), targets.clone(), model.loss_fn)
	torch.cuda.reset_peak_memory_stats()

	# Benchmark
	if global_rank == 0:
		logger.info("Starting benchmark")
	times = []
	for i in range(10):
		model.zero_grad()
		y = pipe(inputs.clone(), targets.clone(), model.loss_fn)
		times.append(pipe.times)

	# Gather memory stats
	mems = [torch.tensor(0.0, device=rank) for _ in range(world_size)] if global_rank == 0 else None
	dist.gather(torch.tensor(torch.cuda.max_memory_allocated() / (2**30), device=rank), mems, 0)

	# Calculate median times
	median_times = medians(times)
	itimes = [{} for _ in range(world_size)] if global_rank == 0 else None
	dist.gather_object(median_times, itimes, 0)

	if global_rank == 0:
		# Log config
		config_dict = {
			"model": args.model,
			"batch_size": args.batch_size,
			"seq_len": args.seq_len,
			"pp_size": pipe.pp,
			"dp_size": pipe.dp,
			"schedule": args.schedule,
			"partitioner": args.partitioner,
		}

		# Get git commit hash
		try:
			git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
			config_dict["git_commit"] = git_commit
		except (subprocess.CalledProcessError, FileNotFoundError):
			config_dict["git_commit"] = "unknown"

		wandb.init(
			project="elf",
			entity="topal-inria",
			job_type="gpt-benchmark",
			config=config_dict,
			mode="offline",
		)

		# Log metrics
		metrics = {
			"total_time": median_times["total"], # total time is the same for all GPUs
		}
		for i, (times, mem) in enumerate(zip(itimes, mems)):
			metrics[f"gpu_{i}_memory_gb"] = mem.item()
			metrics[f"gpu_{i}_idle_time"] = times["idle"]
			metrics[f"gpu_{i}_idle_percentage"] = 100 * times["idle"] / times["total"]

		wandb.log(metrics)
		wandb.finish()

	pipe.clear()

	dist.barrier()
	if dist.is_initialized():
		dist.destroy_process_group()
