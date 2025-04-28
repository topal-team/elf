#!/usr/bin/env python
"""
Profiling script for transformer model performance measurement.

This script measures forward, backward, and weight update times and memory usage
for different transformer model configurations. It collects performance metrics
with and without activation recomputation (checkpointing).

Usage:
    python ilps/profiling.py --config CONFIG_FILE --output OUTPUT_FILE [options]

Arguments:
    --config, -c: JSON file with model hyperparameters
    --output, -o: Output file path for the statistics
    --blocks, -b: Comma-separated list of block sizes to test (default: 1,2,4)
    --batch-sizes, -bs: Comma-separated list of batch sizes to test (default: 1,2,4)
    --iterations, -i: Number of iterations for each configuration (default: 100)
"""

import sys
import json
import argparse
import os

import torch
from torch.utils.checkpoint import checkpoint

sys.path.append(".")

from models.simple import Attention, ChainTransformer
from elf.zb_utils import LayerDW, replace_linear_with_linear_dw
from elf.utils import Timer

# Default model hyperparameters
default_config = {"hidden_dim": 4096, "seq_len": 1024, "num_heads": 32, "dropout": 0.1}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Measure transformer performance with different configurations"
	)
	parser.add_argument("--config", "-c", type=str, help="JSON file with model hyperparameters")
	parser.add_argument(
		"--output",
		"-o",
		type=str,
		default="results/regression_stats.json",
		help="Output file path for the statistics (default: results/regression_stats.json)",
	)
	parser.add_argument(
		"--blocks",
		"-b",
		type=str,
		default="1,2,4",
		help="Comma-separated list of block sizes to test (default: 1,2,4)",
	)
	parser.add_argument(
		"--batch-sizes",
		"-bs",
		type=str,
		default="1,2,4",
		help="Comma-separated list of batch sizes to test (default: 1,2,4)",
	)
	parser.add_argument(
		"--iterations",
		"-i",
		type=int,
		default=100,
		help="Number of iterations to run for each configuration (default: 100)",
	)
	return parser.parse_args()


def load_config(config_path=None):
	config = default_config.copy()

	if config_path:
		try:
			with open(config_path, "r") as f:
				user_config = json.load(f)
			config.update(user_config["model"])
			print(f"Loaded configuration from {config_path}")
		except Exception as e:
			print(f"Error loading config from {config_path}: {e}")
			print("Using default configuration")

	return config


def bparams(model):
	for module in model.modules():
		if isinstance(module, LayerDW):
			module.move_last_computed("input", 0)
			module.move_last_computed("grad_output", 0)
			module.backward(0)


def create_model(n_blocks, config):
	model = ChainTransformer(
		config["hidden_dim"], n_blocks, config["seq_len"], config["num_heads"], config["dropout"]
	).cuda()
	replace_linear_with_linear_dw(model, "cuda:0")
	return model


def measure_times(model, batch_size, n_iter):
	start_mem = torch.cuda.memory_allocated()
	end_mem = torch.cuda.memory_allocated()
	n_blocks = model.num_blocks if hasattr(model, "num_blocks") else len(model.blocks)

	# Calculate parameter memory
	param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
	param_size_per_block = param_size / n_blocks

	print(f"Memory allocated for {n_blocks} blocks: {(end_mem - start_mem) / 1024 / 1024:.3f}MB")
	print(f"Param size per block: {param_size_per_block:.3f}MB / block")

	# Warmup
	sample = model.get_sample(batch_size).cuda()
	for _ in range(10):
		y = model(sample)
		y.sum().backward()
		bparams(model)
	del y
	torch.cuda.synchronize()

	ops = ["F", "B", "W"]
	times = [[], [], []]
	mems = [0, 0, 0]

	for i in range(n_iter):
		sample = model.get_sample(batch_size).cuda()
		start_mem = torch.cuda.memory_allocated()
		with Timer() as t:
			y = model(sample)
		times[0].append(t.time())
		mems[0] = torch.cuda.memory_allocated() - start_mem

		with Timer() as t:
			y.sum().backward()
			del y
		times[1].append(t.time())
		mems[1] = torch.cuda.memory_allocated() - start_mem

		with Timer() as t:
			bparams(model)
		times[2].append(t.time())
		mems[2] = torch.cuda.memory_allocated() - start_mem

	avg_times = []
	avg_mems_per_block_batch = []
	for i in range(3):
		t = 1000 * sum(times[i]) / len(times[i])
		m = mems[i] / 1024 / 1024
		mem_per_block_batch = m / (n_blocks * batch_size)
		print(
			f"{ops[i]}: {t:.3f}ms ({t / n_blocks:.3f} / block), {mem_per_block_batch:.3f}MB / block / batch size"
		)
		avg_times.append(t)
		avg_mems_per_block_batch.append(mem_per_block_batch)

	# Cleanup
	del model
	torch.cuda.empty_cache()

	return avg_times, avg_mems_per_block_batch, param_size_per_block


def measure_memory_only(model, batch_size):
	"""Separate function to measure only Mfp and Mbp without doing timing measurements"""
	n_blocks = len(model.blocks)
	sample = model.get_sample(batch_size).cuda()

	# Warmup for initial param grads allocations
	y = model(sample)
	y.sum().backward()

	# Clear all saved tensors
	del y
	for layer in model.modules():
		if isinstance(layer, LayerDW):
			layer.clear()

	# Measure Mfp (memory for activations)
	start_mem = torch.cuda.memory_allocated()
	y = model(sample)
	y.sum().backward()

	del y

	# Clear gradient outputs while keeping inputs
	for layer in model.modules():
		if isinstance(layer, LayerDW):
			layer.last_grad_output = None

	mfp_val = torch.cuda.memory_allocated() - start_mem
	mfp_per_block_batch = mfp_val / (n_blocks * batch_size) / 1024 / 1024
	print(f"Mfp: {mfp_per_block_batch:.3f}MB")

	# Reset for next measurement
	for layer in model.modules():
		if isinstance(layer, LayerDW):
			layer.clear()

	torch.cuda.empty_cache()
	sample = model.get_sample(batch_size).cuda()

	# Measure Mbp (memory for gradients)
	start_mem = torch.cuda.memory_allocated()
	y = model(sample)
	y.sum().backward()

	del y

	# Clear inputs while keeping gradient outputs
	for layer in model.modules():
		if isinstance(layer, LayerDW):
			layer.last_input = None

	mbp_val = torch.cuda.memory_allocated() - start_mem
	mbp_per_block_batch = mbp_val / (n_blocks * batch_size) / 1024 / 1024
	print(f"Mbp: {mbp_per_block_batch:.3f}MB")

	# Cleanup
	del model
	torch.cuda.empty_cache()

	return mfp_per_block_batch, mbp_per_block_batch


def apply_checkpointing(model):
	"""Apply selective recomputation via checkpointing to attention layers"""
	for name, module in model.named_modules():
		if isinstance(module, Attention):
			original_forward = getattr(module, "forward")

			def wrapped_forward(*args, **kwargs):
				return checkpoint(original_forward, *args, **kwargs, use_reentrant=False)

			setattr(module, "forward", wrapped_forward)
	return model


def main():
	args = parse_args()
	config = load_config(args.config)

	# Parse block sizes and batch sizes from command line
	block_sizes = [int(x) for x in args.blocks.split(",")]
	batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
	n_iter = args.iterations

	print("Using model configuration:")
	for key, value in config.items():
		print(f"  {key}: {value}")

	print("Testing with:")
	print(f"  block_sizes: {block_sizes}")
	print(f"  batch_sizes: {batch_sizes}")
	print(f"  iterations: {n_iter}")

	stats = {
		"no_recompute": {"times": [], "features": [], "memory": [], "param_size_per_block": []},
		"recompute": {"times": [], "features": [], "memory": [], "param_size_per_block": []},
		"mfp_per_block_batch": 0,
		"mbp_per_block_batch": 0,
	}

	# First, collect Mfp and Mbp metrics (just once, as they are normalized per block and batch)
	print("\nMeasuring Mfp and Mbp metrics:")
	# Use the first block size and batch size for measurement
	n = block_sizes[0]
	bs = batch_sizes[0]
	print(f"\nMeasuring memory for {n} blocks, batch size {bs}:")
	model = create_model(n, config)
	mfp, mbp = measure_memory_only(model, bs)
	stats["mfp_per_block_batch"] = mfp
	stats["mbp_per_block_batch"] = mbp

	# Then measure performance without recomputation
	print("\nMeasuring performance without recomputation:")
	for n in block_sizes:
		for bs in batch_sizes:
			print(f"\nMeasuring for {n} blocks, batch size {bs}:")
			try:
				model = create_model(n, config)
				times, mems, param_size_per_block = measure_times(model, bs, n_iter)

				stats["no_recompute"]["times"].append(times)
				stats["no_recompute"]["features"].append([n, bs])
				stats["no_recompute"]["memory"].append(mems)
				stats["no_recompute"]["param_size_per_block"].append(param_size_per_block)
			except torch.cuda.OutOfMemoryError:
				print(f"CUDA out of memory for {n} blocks, batch size {bs}, skipping")

	print("\n-- With selective recomputation --\n")

	for n in block_sizes:
		for bs in batch_sizes:
			print(f"\nMeasuring recompute for {n} blocks, batch size {bs}:")
			try:
				model = create_model(n, config)
				model = apply_checkpointing(model)
				times, mems, param_size_per_block = measure_times(model, bs, n_iter)
				stats["recompute"]["times"].append(times)
				stats["recompute"]["features"].append([n, bs])
				stats["recompute"]["memory"].append(mems)
				stats["recompute"]["param_size_per_block"].append(param_size_per_block)
			except torch.cuda.OutOfMemoryError:
				print(f"CUDA out of memory for {n} blocks, batch size {bs}, skipping")

	# Save stats to file
	output_path = args.output
	print(f"Saving results to {output_path}")

	# Ensure the directory exists
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	with open(output_path, "w") as f:
		json.dump(stats, f)


if __name__ == "__main__":
	main()
