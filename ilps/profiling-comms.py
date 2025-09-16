#!/usr/bin/env python
"""
Communication profiling for distributed model execution.

Measures average communication time (send/recv) for a tensor shaped like a stage
boundary activation and writes the scalar Tcomm into all stages of the config.

Usage:
    torchrun --nproc-per-node=2 ilps/profiling-comms.py --config-file CONFIG_JSON [options]
"""

import os
import sys
import json
import argparse
import traceback

import torch
import torch.distributed as dist

sys.path.append(".")

from models.utils import add_transformer_args, model_config_from_args


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------


def parse_args():
	parser = argparse.ArgumentParser(description="Measure communication time between GPUs")
	# Script-specific arguments
	parser.add_argument(
		"--output",
		type=str,
		help="Path to save the updated configuration file (defaults to overwriting config_file)",
	)
	parser.add_argument(
		"--microbatch_size", type=int, default=1, help="Microbatch size for communication (default: 1)"
	)
	parser.add_argument(
		"--iterations", type=int, default=100, help="Number of iterations for timing (default: 100)"
	)

	# Add model hyper-parameter flags (provides --config-file, etc.)
	add_transformer_args(parser, model_type="full")

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

	# ------------------------------------------------------------------
	# Build model configuration using shared helper
	# ------------------------------------------------------------------
	config = model_config_from_args(args, model_type="full")
	hidden_size = config["hidden_dim"]
	seq_len = config["seq_len"]
	dtype = config["dtype"]
	print("Using model configuration from CLI / config file:")
	print(f"  hidden_dim: {hidden_size}")
	print(f"  seq_len: {seq_len}")
	print(f"  dtype: {dtype}")

	# Load JSON config (if provided) for later update of Tcomm
	config_file = args.config_file or "ilps/configs/default.json"
	try:
		with open(config_file, "r") as f:
			config_json = json.load(f)
	except Exception:
		config_json = None

	rank = int(os.environ.get("RANK", "0"))
	local_rank = int(os.environ.get("LOCAL_RANK", "0"))

	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

	microbatch_size = args.microbatch_size

	x = torch.randn(microbatch_size, seq_len, hidden_size, dtype=dtype, device="cuda")
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
			end.synchronize()
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
			end.synchronize()
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

		# Update configuration file if we managed to read it
		if config_json is not None:
			output = args.output if args.output else config_file

			try:
				for stage in config_json["stages"]:
					stage["Tcomm"] = avg_comm

				os.makedirs(os.path.dirname(output), exist_ok=True)
				with open(output, "w") as f:
					json.dump(config_json, f, indent=2)

				print(f"Updated Tcomm value ({avg_comm:.6f} ms) in {output}")
			except Exception as e:
				print(f"Error updating configuration file: {e}")
				print(traceback.format_exc())
		else:
			print("No configuration JSON provided/read; skipping Tcomm update.")

	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
