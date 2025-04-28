#!/usr/bin/env python
"""
Communication profiling script for distributed model execution.

This script measures the time required for communication between GPUs in a
distributed setting. It performs send/receive operations to calculate average
communication time and updates the configuration file with the Tcomm parameter.

Usage:
    torchrun --nproc-per-node=2 ilps/profiling-comms.py --config CONFIG_FILE [options]

Arguments:
    --config_file: Path to the configuration file to update (default: ilps/configs/default.json)
    --output_file: Path to save the updated configuration file (defaults to overwriting config_file)
    --microbatch_size: Microbatch size for communication (default: 2)
    --iterations: Number of iterations for timing (default: 1000)

Note: This script must be run with at least 2 GPUs using torchrun or a similar launcher.
"""

import os
import json
import argparse
import sys
import torch
import torch.distributed as dist


def parse_args():
	parser = argparse.ArgumentParser(description="Measure communication time between GPUs")
	parser.add_argument(
		"--config_file",
		type=str,
		default="ilps/configs/default.json",
		help="Path to the configuration file to update",
	)
	parser.add_argument(
		"--output_file",
		type=str,
		help="Path to save the updated configuration file (defaults to overwriting config_file)",
	)
	parser.add_argument(
		"--microbatch_size", type=int, default=2, help="Microbatch size for communication (default: 2)"
	)
	parser.add_argument(
		"--iterations", type=int, default=1000, help="Number of iterations for timing (default: 1000)"
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

	# Read the configuration file to get model parameters
	config_file = args.config_file
	try:
		with open(config_file, "r") as f:
			config = json.load(f)

		# Get hidden size and sequence length from config
		hidden_size = config["model"]["hidden_dim"]
		seq_len = config["model"]["seq_len"]
		print(f"Using model configuration from {config_file}:")
		print(f"  hidden_dim: {hidden_size}")
		print(f"  seq_len: {seq_len}")
	except Exception as e:
		print(f"Error reading configuration file: {e}")
		print("Using default values")
		hidden_size = 1024
		seq_len = 256

	rank = int(os.environ.get("RANK", "0"))
	local_rank = int(os.environ.get("LOCAL_RANK", "0"))

	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	microbatch_size = args.microbatch_size

	x = torch.randn(microbatch_size, seq_len, hidden_size, device="cuda")
	# Ensure all GPUs are synchronized before starting
	dist.barrier()
	torch.cuda.synchronize()

	# Send to next rank in ring (or back to 0 if last rank)
	dst = (rank + 1) % world_size
	src = (rank - 1) % world_size

	# Warmup iterations
	for _ in range(3):
		if rank % 2 == 0:
			dist.send(x, dst)
		else:
			dist.recv(x, src)
		torch.cuda.synchronize()

	# Measure send time
	n_iters = args.iterations
	send_times = []
	recv_times = []

	for i in range(n_iters):
		torch.cuda.synchronize()
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)

		# Measure send
		if rank % 2 == 0:
			start.record()
			dist.send(x, dst)
			end.record()
			torch.cuda.synchronize()
			send_times.append(start.elapsed_time(end))
		else:
			dist.recv(x, src)

	dist.barrier()

	# Now measure receive
	for i in range(n_iters):
		torch.cuda.synchronize()
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)

		if rank % 2 == 0:
			start.record()
			dist.recv(x, src)
			end.record()
			torch.cuda.synchronize()
			recv_times.append(start.elapsed_time(end))
		else:
			dist.send(x, dst)

	if rank == 0:
		avg_send = sum(send_times) / len(send_times)
		avg_recv = sum(recv_times) / len(recv_times)
		avg_comm = (avg_send + avg_recv) / 2  # Average of send and receive times

		print(f"Average send time: {avg_send:.3f} ms")
		print(f"Average recv time: {avg_recv:.3f} ms")
		print(f"Average communication time: {avg_comm:.3f} ms")
		print(f"Message size: {x.element_size() * x.nelement() / 1024 / 1024:.2f} MB")

		# Update configuration file
		output_file = args.output_file if args.output_file else config_file

		try:
			config["Tcomm"] = avg_comm

			os.makedirs(os.path.dirname(output_file), exist_ok=True)
			with open(output_file, "w") as f:
				json.dump(config, f, indent=2)

			print(f"Updated Tcomm value ({config['Tcomm']:.6f} s) in {output_file}")
		except Exception as e:
			print(f"Error updating configuration file: {e}")

	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
