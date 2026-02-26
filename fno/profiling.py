#!/usr/bin/env python
"""
Per-stage profiling for partitioned FNO models.

Profiles each stage of a partitioned FNO individually and emits raw
per-stage statistics without linear regression or scaling. These raw stats can
be consumed directly by the ILPs.

Measured per stage (full recompute only):
  - Times: [Tf, Tb, Tw] in milliseconds
  - Post-op memory deltas: [Mf, Mb, Mw] in MB
  - Peak memory per op: [PeakF, PeakB, PeakW] in MB
  - Parameter size: Mparams in MB

Additionally, we measure per-stage Mfp and Mbp.

Usage:
	python fno/profile.py --output OUTPUT_FILE --nstages 4 [model options]
"""

import json
import argparse
import os
import torch

from neuralop.layers.embeddings import Embedding, GridEmbeddingND

from elf.utils import Timer
from elf.zb_utils import LayerDW, replace_layer_with_layer_dw
from fno.benchmark import build_fno_from_config, build_fno_stages
from benchmarks.benchmark_utils import meta_to_device, balanced_partition


def parse_args():
	parser = argparse.ArgumentParser(description="Per-stage profiling for partitioned FNO")
	parser.add_argument(
		"--config-file",
		"-c",
		type=str,
		default=None,
		help="Path to JSON config file (overrides defaults, CLI args override config)",
	)
	parser.add_argument(
		"--output", "-o", type=str, default=None, help="Output file path for the statistics"
	)
	parser.add_argument(
		"--nstages",
		"-n",
		type=int,
		default=None,
		help="Number of partitions/stages (defaults to ngpus from config)",
	)
	parser.add_argument("--batch-size", "-bs", type=int, default=1, help="Batch size for profiling")
	parser.add_argument(
		"--iterations", "-i", type=int, default=5, help="Number of iterations for timing averages"
	)

	# FNO model hyperparameters (all optional, can come from config file)
	parser.add_argument("--n-modes", type=int, nargs="+", default=None, help="Fourier modes")
	parser.add_argument("--in-channels", type=int, default=None, help="Input channels")
	parser.add_argument("--out-channels", type=int, default=None, help="Output channels")
	parser.add_argument("--hidden-channels", type=int, default=None, help="Hidden channels")
	parser.add_argument("--n-layers", type=int, default=None, help="Number of FNO layers")
	parser.add_argument(
		"--projection-channel-ratio", type=float, default=None, help="Projection channel ratio"
	)

	# Data shape (excluding batch dimension)
	parser.add_argument(
		"--spatial-dims",
		nargs="+",
		default=None,
		help="Spatial dimensions of input (height, width, ...)",
	)

	# Complex data option
	parser.add_argument("--complex-data", action="store_true", default=None, help="Use complex data")
	parser.add_argument("--no-complex", action="store_true", help="Disable complex data")

	return parser.parse_args()


def build_config(args) -> tuple[dict, int, str]:
	"""Build final config by merging config file and CLI args.

	The returned config uses the same nested structure as the JSON config files
	(with "model" and "data" sections), so it can be passed directly to
	build_fno_from_config.

	Returns (config, nstages, output_path).
	"""
	config = {"model": {}, "data": {}}
	nstages = 4
	output_path = "results/fno_stage_stats.json"

	if args.config_file:
		with open(args.config_file, "r") as f:
			config = json.load(f)
		nstages = config.get("ngpus", nstages)

	# CLI args override config file values
	model = config.setdefault("model", {})
	data = config.setdefault("data", {})

	if args.n_modes is not None:
		model["n_modes"] = args.n_modes
	if args.in_channels is not None:
		model["in_channels"] = args.in_channels
	if args.out_channels is not None:
		model["out_channels"] = args.out_channels
	if args.hidden_channels is not None:
		model["hidden_channels"] = args.hidden_channels
	if args.n_layers is not None:
		model["n_layers"] = args.n_layers
	if args.projection_channel_ratio is not None:
		model["projection_channel_ratio"] = args.projection_channel_ratio
	if args.spatial_dims is not None:
		data["spatial_dims"] = [int(d) for d in args.spatial_dims]
	if args.complex_data is not None:
		data["complex"] = args.complex_data
	if args.no_complex:
		data["complex"] = False
	if args.nstages is not None:
		nstages = args.nstages
	if args.output is not None:
		output_path = args.output

	return config, nstages, output_path


def _stage_bparams(stage: torch.nn.Module):
	for module in stage.modules():
		if isinstance(module, LayerDW):
			module.move_last_computed("input", 0)
			module.move_last_computed("grad_output", 0)
			# FNO sub-blocks have all modules registered as theirs ; we need to check that this module is actually part of the computation for this block
			if module.is_empty("input") or module.is_empty("grad_output"):
				continue
			module.backward(0)


def _stage_clear(stage: torch.nn.Module):
	for module in stage.modules():
		if isinstance(module, LayerDW):
			module.clear()


def _get_stage_sample(
	stage: torch.nn.Sequential, batch_size: int, dtype: torch.dtype, config: dict
):
	"""Generate appropriate input for a stage based on its first module."""
	hidden_channels = config["model"]["hidden_channels"]
	in_channels = config["model"]["in_channels"]
	spatial_dims = config["data"]["spatial_dims"]

	first = stage[0]

	# First stage gets raw input (positional embedding or lifting)
	if isinstance(first, (Embedding, GridEmbeddingND)):
		shape = (batch_size, in_channels, *spatial_dims)
		return torch.randn(shape, dtype=dtype, device="cuda")
	else:
		# Intermediate stages get hidden representations
		shape = (batch_size, hidden_channels, *spatial_dims)
		return torch.randn(shape, dtype=dtype, device="cuda")


def _measure_stage_times(stage: torch.nn.Module, sample, target, n_iter: int, loss_fn):
	ops = ["f", "b", "w"]
	times = {k: [] for k in ops}
	mems = {k: [] for k in ops}
	peaks = {k: [] for k in ops}

	# Warmup & initial param grads allocations
	y = stage(sample)
	loss_fn(y, target).backward()
	_stage_bparams(stage)
	del y
	torch.cuda.synchronize()

	for _ in range(n_iter):
		# Forward
		torch.cuda.reset_peak_memory_stats()
		start_mem = torch.cuda.memory_allocated()
		with Timer() as t:
			x = (
				sample.clone()
				.detach()
				.requires_grad_(torch.is_floating_point(sample) or sample.is_complex())
			)
			y = stage(x)
			loss = loss_fn(y, target)
		torch.cuda.synchronize()
		times["f"].append(1000 * t.time())
		mems["f"].append((torch.cuda.memory_allocated() - start_mem) / 1024 / 1024)
		peaks["f"].append((torch.cuda.max_memory_allocated() - start_mem) / 1024 / 1024)

		# Backward
		torch.cuda.reset_peak_memory_stats()
		before_b = torch.cuda.memory_allocated()
		with Timer() as t:
			loss.backward()
			del x, y, loss
		torch.cuda.synchronize()
		times["b"].append(1000 * t.time())
		mems["b"].append((torch.cuda.memory_allocated() - start_mem) / 1024 / 1024)
		peaks["b"].append((torch.cuda.max_memory_allocated() - before_b) / 1024 / 1024)

		# Weight update
		torch.cuda.reset_peak_memory_stats()
		before_w = torch.cuda.memory_allocated()
		with Timer() as t:
			_stage_bparams(stage)
		torch.cuda.synchronize()

		times["w"].append(1000 * t.time())
		mems["w"].append((torch.cuda.memory_allocated() - start_mem) / 1024 / 1024)
		peaks["w"].append((torch.cuda.max_memory_allocated() - before_w) / 1024 / 1024)

	avg_times = [float(sum(times[k]) / len(times[k])) for k in ops]
	avg_mems = [float(sum(mems[k]) / len(mems[k])) for k in ops]
	avg_peaks = [float(sum(peaks[k]) / len(peaks[k])) for k in ops]
	return avg_times, avg_mems, avg_peaks


def _measure_stage_mfp_mbp(stage: torch.nn.Module, sample) -> tuple[float, float]:
	"""Measure Mfp and Mbp (in MB) for a single stage.

	- Mfp: memory kept by activations after backward (inputs kept, grad_outputs cleared)
	- Mbp: memory kept by gradients after forward (grad_outputs kept, inputs cleared)
	"""
	# Warmup for initial param grads allocations
	y = stage(sample)

	if y.is_complex():
		y = torch.view_as_real(y)
	y.sum().backward()

	# Clear all saved tensors
	del y
	_stage_clear(stage)

	# Measure Mfp
	start_mem = torch.cuda.memory_allocated()
	y = stage(sample)
	if y.is_complex():
		y = torch.view_as_real(y)
	y.sum().backward()
	del y
	for layer in stage.modules():
		if isinstance(layer, LayerDW):
			layer.last_grad_output = None
	mfp_mb = (torch.cuda.memory_allocated() - start_mem) / 1024 / 1024

	# Reset
	_stage_clear(stage)

	# Mbp
	start_mem = torch.cuda.memory_allocated()
	y = stage(sample)
	if y.is_complex():
		y = torch.view_as_real(y)
	y.sum().backward()
	del y
	for layer in stage.modules():
		if isinstance(layer, LayerDW):
			layer.last_input = None
	mbp_mb = (torch.cuda.memory_allocated() - start_mem) / 1024 / 1024

	_stage_clear(stage)
	return float(mfp_mb), float(mbp_mb)


def _measure_peak_no_grad(stage: torch.nn.Module, sample):
	"""Measure the peak memory of a stage without gradients"""
	torch.cuda.reset_peak_memory_stats()
	x = sample.clone().detach()
	start_mem = torch.cuda.memory_allocated()
	with torch.no_grad():
		y = stage(x)  # noqa: F841
	torch.cuda.synchronize()
	return float((torch.cuda.max_memory_allocated() - start_mem) / 1024 / 1024)


def _measure_stage_input_output_mem(stage: torch.nn.Module, sample):
	"""Measure the input and output memory of a stage"""
	y = stage(sample)
	# For complex tensors, nbytes accounts for both real and imaginary parts
	return sample.nbytes / (1024 * 1024), y.nbytes / (1024 * 1024)


def _measure_params_memory(stage: torch.nn.Module, sample: torch.Tensor):
	"""Measure the memory of the parameters of a stage
	We don't rely on parameters() because the partition may retain all submodules registrations,
	even if they are not part of the stage.
	"""
	stage.zero_grad(set_to_none=True)
	y = stage(sample)
	if y.is_complex():
		y = torch.view_as_real(y)
	y.sum().backward()
	return sum(p.nbytes for p in list(stage.parameters()) if p.grad is not None) / 1024 / 1024


def main():
	args = parse_args()
	config, nstages, output_path = build_config(args)

	print("Using model configuration:")
	for key, value in config["model"].items():
		print(f"  {key}: {value}")
	print("Data configuration:")
	for key, value in config["data"].items():
		print(f"  {key}: {value}")
	print(f"nstages: {nstages}")
	print(f"batch_size: {args.batch_size}")
	print(f"iterations: {args.iterations}")

	# Instantiate full model on meta device
	with torch.device("meta"):
		model, dtype = build_fno_from_config(config)

	# Partition into stages
	factors = balanced_partition(model.fno_blocks.n_layers, list(range(nstages)))
	stages = build_fno_stages(model, factors)

	# Replace supported layers with LayerDW for W update simulation
	for s in stages:
		replace_layer_with_layer_dw(s)

	# Measure per stage
	staged_stats = []

	# Dedup identical stages by structural signature
	def stage_signature(stage: torch.nn.Module) -> tuple:
		mods = []
		for name, mod in stage.named_modules():
			if name == "":
				continue
			mods.append(type(mod).__name__)
		params = [(tuple(p.shape), p.dtype) for p in stage.parameters(recurse=True)]
		return (tuple(mods), tuple(params))

	sig_to_stats: dict[tuple, dict] = {}
	for idx, stage in enumerate(stages):
		print(f"\nProfiling stage {idx}/{len(stages) - 1}")
		sig = stage_signature(stage)
		if sig in sig_to_stats:
			print("Identical stage detected; reusing measured stats.")
			staged_stats.append(sig_to_stats[sig])
			continue

		stage = meta_to_device(stage)
		sample = _get_stage_sample(stage, args.batch_size, dtype, config)

		# For last stage, create target matching output shape
		if idx == len(stages) - 1:
			target_shape = (
				args.batch_size,
				config["model"]["out_channels"],
				*config["data"]["spatial_dims"],
			)
			target = torch.randn(target_shape, dtype=dtype, device="cuda")
		else:
			target = None

		loss_fn = (
			torch.nn.functional.l1_loss
			if idx == len(stages) - 1
			else lambda x, y: torch.view_as_real(x).sum() if x.is_complex() else x.sum()
		)

		# Mfp/Mbp for backward remat options
		mfp_mb, mbp_mb = _measure_stage_mfp_mbp(stage, sample)
		peak_no_grad = _measure_peak_no_grad(stage, sample)
		input_mem, output_mem = _measure_stage_input_output_mem(stage, sample)

		# No recompute measurements
		times_nr, mems_nr, peaks_nr = _measure_stage_times(
			stage, sample, target, args.iterations, loss_fn
		)

		print(f"Times (ms): F={times_nr[0]:.2f}, B={times_nr[1]:.2f}, W={times_nr[2]:.2f}")
		print(f"Kept (MB): F={mems_nr[0]:.2f}, B={mems_nr[1]:.2f}, W={mems_nr[2]:.2f}")
		print(f"Peaks (MB): F={peaks_nr[0]:.2f}, B={peaks_nr[1]:.2f}, W={peaks_nr[2]:.2f}")

		# Parameter memory in MB for stage
		mparams_mb = _measure_params_memory(stage, sample)

		stage_cfg = {
			"T": times_nr,  # [Tf, Tb, Tw]
			"M": mems_nr,  # [Mf, Mb, Mw]
			"Mpeak": peaks_nr,  # [PeakF, PeakB, PeakW]
			"Mparams": float(mparams_mb),
			"Mpeak_no_grad": float(peak_no_grad),
			"Tcomm": 0.0,  # to be filled by profile_comms.py
			"forward_remat_options": [],  # full remat only, no selective options
			"backward_remat_options": [
				{
					"name": "activations_bwd",
					"overhead": float(times_nr[0]),  # f
					"mem_freed": float(mfp_mb) - float(input_mem),
				}
			],
			"Mfp": float(mfp_mb),
			"Mbp": float(mbp_mb),
			"Minput": float(input_mem),
			"Moutput": float(output_mem),
		}
		staged_stats.append(stage_cfg)
		sig_to_stats[sig] = stage_cfg

		_stage_clear(stage)
		stage.to("meta")

	# Save results
	print(f"\nSaving results to {output_path}")

	result = {
		"model": config["model"],
		"data": config["data"],
		"stages": staged_stats,
		"nparams": sum(p.numel() for p in model.parameters()),
	}

	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	with open(output_path, "w") as f:
		json.dump(result, f, indent=2)


if __name__ == "__main__":
	main()
