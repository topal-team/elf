#!/usr/bin/env python
"""
Communication profiling for FNO distributed model execution.

Measures average communication time (send/recv) for a tensor shaped like a stage
boundary activation and writes the scalar Tcomm into all stages of the config.

Usage:
    torchrun --nproc-per-node=2 fno/profile_comms.py --config-file CONFIG_JSON [options]
"""

import os
import sys
import json
import argparse
import traceback

import torch
import torch.distributed as dist


def parse_args():
	parser = argparse.ArgumentParser(description="Measure communication time between GPUs for FNO")
	parser.add_argument(
		"--config-file",
		"-c",
		type=str,
		required=True,
		help="Path to the profiled config file with model and stages info",
	)
	parser.add_argument(
		"--output",
		type=str,
		help="Path to save the updated configuration file (defaults to overwriting config_file)",
	)
	parser.add_argument(
		"--microbatch-size", type=int, default=1, help="Microbatch size for communication (default: 1)"
	)
	parser.add_argument(
		"--iterations", type=int, default=100, help="Number of iterations for timing (default: 100)"
	)
	return parser.parse_args()


def main():
	args = parse_args()

	# Check for multiple GPUs immediately
	world_size = int(os.environ.get("WORLD_SIZE", "1"))
	if world_size <= 1:
		print("Error: Communication profiling requires multiple GPUs.")
		print("This script should be run with torchrun or similar to use multiple GPUs.")
		print("Exiting without making changes to the configuration file.")
		sys.exit(1)

	# Load config
	try:
		with open(args.config_file, "r") as f:
			config_json = json.load(f)
	except Exception as e:
		print(f"Error loading config file: {e}")
		sys.exit(1)

	# Extract model config
	model_config = config_json.get("model", {})
	data_config = config_json.get("data", {})
	hidden_channels = model_config.get("hidden_channels")
	spatial_dims = tuple(data_config.get("spatial_dims", [1024, 256]))
	complex_data = data_config.get("complex", True)
	dtype = torch.complex64 if complex_data else torch.float32

	if hidden_channels is None:
		print("Error: 'hidden_channels' not found in model config")
		sys.exit(1)

	print("Using model configuration from config file:")
	print(f"  hidden_channels: {hidden_channels}")
	print(f"  spatial_dims: {spatial_dims}")
	print(f"  complex_data: {complex_data}")
	print(f"  dtype: {dtype}")

	rank = int(os.environ.get("RANK", "0"))
	local_rank = int(os.environ.get("LOCAL_RANK", "0"))

	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

	microbatch_size = args.microbatch_size

	# Create tensor with FNO activation shape: (batch, channels, *spatial_dims)
	shape = (microbatch_size, hidden_channels, *spatial_dims)
	x = torch.randn(shape, dtype=dtype, device="cuda")

	print(f"Rank {rank}: Testing communication with tensor shape {x.shape}")

	# Ensure all GPUs are synchronized before starting
	dist.barrier()
	torch.cuda.synchronize()

	# Send to next rank in ring (or back to 0 if last rank)
	dst = (rank + 1) % world_size
	src = (rank - 1) % world_size

	# Warmup iterations
	for _ in range(3):
		if rank % 2 == 0:
			dist.send(x, dst=dst)
			dist.recv(x, src=src)
		else:
			dist.recv(x, src=src)
			dist.send(x, dst=dst)
	dist.barrier()
	torch.cuda.synchronize()

	# Timed iterations
	times = []
	for _ in range(args.iterations):
		torch.cuda.synchronize()
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)

		start.record()
		if rank % 2 == 0:
			dist.send(x, dst=dst)
			dist.recv(x, src=src)
		else:
			dist.recv(x, src=src)
			dist.send(x, dst=dst)
		end.record()

		torch.cuda.synchronize()
		times.append(start.elapsed_time(end))

	dist.barrier()

	# Average time across all ranks
	avg_time_ms = sum(times) / len(times)
	all_avg_times = [torch.zeros(1, device="cuda") for _ in range(world_size)]
	dist.all_gather(all_avg_times, torch.tensor([avg_time_ms], device="cuda"))

	# Use global average
	global_avg_time_ms = sum(t.item() for t in all_avg_times) / len(all_avg_times)

	if rank == 0:
		print(f"\nCommunication time: {global_avg_time_ms:.3f} ms")
		print(f"  (averaged over {args.iterations} iterations)")

		# Update Tcomm in all stages
		if "stages" in config_json:
			for stage in config_json["stages"]:
				stage["Tcomm"] = global_avg_time_ms
			print(f"  Updated Tcomm in {len(config_json['stages'])} stages")

		# Save updated config
		output_file = args.output or args.config_file
		try:
			with open(output_file, "w") as f:
				json.dump(config_json, f, indent=2)
			print(f"\nSaved updated config to: {output_file}")
		except Exception as e:
			print(f"Error saving config: {e}")
			traceback.print_exc()
			sys.exit(1)

	dist.destroy_process_group()


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"Error: {e}")
		traceback.print_exc()
		sys.exit(1)
