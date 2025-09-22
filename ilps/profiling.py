#!/usr/bin/env python
"""
Per-stage profiling for partitioned FullTransformer models.

Profiles each stage of a partitioned FullTransformer individually and emits raw
per-stage statistics without linear regression or scaling. These raw stats can
be consumed directly by the ILPs.

Measured per stage (no recompute and selective forward recompute):
  - Times: [Tf, Tb, Tw] in milliseconds
  - Post-op memory deltas: [Mf, Mb, Mw] in MB
  - Peak memory per op: [PeakF, PeakB, PeakW] in MB
  - Parameter size: Mparams in MB
Additionally, we measure per-stage Mfp and Mbp.

Usage:
    python ilps/profiling.py --output OUTPUT_FILE --nstages 4 [model options]
"""

import json
import argparse
import os
import torch
from torch.utils.checkpoint import checkpoint

from models.simple import Attention
from models.utils import add_transformer_args, model_config_from_args, build_model_from_args
from elf.zb_utils import LayerDW, replace_linear_with_linear_dw
from elf.utils import Timer
from benchmarks.benchmark_utils import meta_to_device

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------


def parse_args():
	parser = argparse.ArgumentParser(
		description="Per-stage profiling for partitioned FullTransformer"
	)
	# Script-specific arguments
	parser.add_argument(
		"--output",
		"-o",
		type=str,
		default="results/stage_stats.json",
		help="Output file path for the statistics (default: results/stage_stats.json)",
	)
	parser.add_argument(
		"--nstages",
		"-n",
		type=int,
		default=4,
		help="Number of partitions/stages to split the model into (default: 4)",
	)
	parser.add_argument(
		"--batch-size", "-bs", type=int, default=1, help="Batch size for profiling (default: 1)"
	)
	parser.add_argument(
		"--iterations",
		"-i",
		type=int,
		default=5,
		help="Number of iterations for timing averages (default: 5)",
	)

	# Add model hyper-parameter flags from utils (includes --sdp-backend, --config-file, ...)
	add_transformer_args(parser)

	return parser.parse_args()


def _partition_full_transformer(model, nstages: int):
	"""Create a simple sequential partition of a FullTransformer into nstages.

	The first stage wraps the embedding + a slice of blocks, the last stage wraps
	the remaining blocks + output head, intermediate stages contain only blocks.
	"""
	assert len(model.blocks) >= nstages, "nstages must be <= number of blocks"
	num_blocks = len(model.blocks)
	base = num_blocks // nstages
	rem = num_blocks % nstages

	parts = []
	start = 0
	for i in range(nstages):
		count = base + (1 if i < rem else 0)
		end = start + count
		modules = []
		if i == 0:
			modules.append(model.embed)
		modules.extend(model.blocks[start:end])
		if i == nstages - 1:
			modules.append(model.head)
		parts.append(torch.nn.Sequential(*modules))
		start = end
	return parts


def _stage_bparams(stage: torch.nn.Module):
	for module in stage.modules():
		if isinstance(module, LayerDW):
			module.move_last_computed("input", 0)
			module.move_last_computed("grad_output", 0)
			module.backward(0)


def _stage_clear(stage: torch.nn.Module):
	for module in stage.modules():
		if isinstance(module, LayerDW):
			module.clear()


def _get_stage_sample(
	stage: torch.nn.Sequential, batch_size: int, dtype: torch.dtype, hidden_dim: int, seq_len: int
):
	# If the first submodule is embedding, feed token ids; otherwise feed hidden states
	first = stage[0]
	if isinstance(first, torch.nn.Embedding):
		return torch.randint(
			0, first.num_embeddings, (batch_size, seq_len), dtype=torch.int64, device="cuda"
		)
	else:
		return torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device="cuda")


def _measure_stage_times(stage: torch.nn.Module, sample, target, n_iter: int, loss_fn):
	ops = ["f", "b", "w"]
	times = {k: [] for k in ops}
	mems = {k: [] for k in ops}
	peaks = {k: [] for k in ops}

	# Warmup
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
			x = sample.clone()
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
	y.sum().backward()

	# Clear all saved tensors
	del y
	for layer in stage.modules():
		if isinstance(layer, LayerDW):
			layer.clear()

	# Measure Mfp
	start_mem = torch.cuda.memory_allocated()
	y = stage(sample)
	y.sum().backward()
	del y
	for layer in stage.modules():
		if isinstance(layer, LayerDW):
			layer.last_grad_output = None
	mfp_mb = (torch.cuda.memory_allocated() - start_mem) / 1024 / 1024

	# Reset
	for layer in stage.modules():
		if isinstance(layer, LayerDW):
			layer.clear()
	torch.cuda.empty_cache()

	# Mbp
	start_mem = torch.cuda.memory_allocated()
	y = stage(sample)
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
	start_mem = torch.cuda.memory_allocated()
	with torch.no_grad():
		x = sample.clone()
		y = stage(x) # noqa: F841
	torch.cuda.synchronize()
	return float((torch.cuda.max_memory_allocated() - start_mem) / 1024 / 1024)


def _apply_checkpointing(stage: torch.nn.Module):
	"""Apply selective recomputation via checkpointing to attention layers in a stage"""
	for _, module in stage.named_modules():
		if isinstance(module, Attention):
			original_forward = getattr(module, "forward")

			def wrapped_forward(*args, **kwargs):
				return checkpoint(original_forward, *args, **kwargs, use_reentrant=True)

			setattr(module, "forward", wrapped_forward)
	return stage


def main():
	args = parse_args()

	# Build the model configuration using the shared helper
	config = model_config_from_args(args)
	dtype = config.get("dtype")
	hidden_dim = config["hidden_dim"]
	seq_len = config["seq_len"]

	print("Using model configuration:")
	for key, value in config.items():
		print(f"  {key}: {value}")
	print(f"nstages: {args.nstages}")
	print(f"batch_size: {args.batch_size}")
	print(f"iterations: {args.iterations}")

	# Instantiate full model once on CUDA
	with torch.device("meta"):
		model, dtype = build_model_from_args(args)
	# Partition into stages (simple balanced split)
	stages = _partition_full_transformer(model, args.nstages)
	# Replace linears with LayerDW for each stage for W update simulation
	for s in stages:
		replace_linear_with_linear_dw(s, "meta")

	# Measure per stage
	staged_stats = []

	# Dedup identical stages by structural signature (parameter shapes and module types)
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
		sample = _get_stage_sample(stage, args.batch_size, dtype, hidden_dim, seq_len)
		target = model.get_target(args.batch_size, dtype, "cuda") if idx == len(stages) - 1 else None

		# Loss function for last stage, otherwise dummy loss function
		loss_fn = model.loss_fn if idx == len(stages) - 1 else lambda x, y: x.sum()

		# Mfp/Mbp for backward remat options
		mfp_mb, mbp_mb = _measure_stage_mfp_mbp(stage, sample)
		peak_no_grad = _measure_peak_no_grad(stage, sample)

		# No recompute
		times_nr, mems_nr, peaks_nr = _measure_stage_times(
			stage, sample, target, args.iterations, loss_fn
		)
		print(f"Peaks without recompute: {peaks_nr}")

		# SR on forward (attention only)
		stage_sr = _apply_checkpointing(stage)
		times_sr, mems_sr, peaks_sr = _measure_stage_times(
			stage_sr, sample, target, args.iterations, loss_fn
		)
		print(f"Peaks with recompute: {peaks_sr}")

		# Parameter memory in MB for stage
		mparams_mb = sum(p.numel() * p.element_size() for p in stage.parameters()) / 1024 / 1024

		# Forward SR: memory freed and overhead
		sr_mem_freed = max(mems_nr[0] - mems_sr[0], 0.0)
		sr_overhead = max(times_sr[1] - times_nr[1], 0.0)  # additional backward time

		stage_cfg = {
			"T": times_nr,  # [Tf, Tb, Tw]
			"M": mems_nr,  # [Mf, Mb, Mw]
			"Mpeak": peaks_nr,  # [PeakF, PeakB, PeakW]
			"Mparams": float(mparams_mb),
			"Mpeak_no_grad": float(peak_no_grad),
			"Tcomm": 0.0,  # to be filled by profiling-comms.py
			"forward_remat_options": [
				{"name": "selective_fwd", "overhead": float(sr_overhead), "mem_freed": float(sr_mem_freed)}
			],
			"backward_remat_options": [
				{"name": "activations_bwd", "overhead": float(times_nr[0]), "mem_freed": float(mfp_mb)}
			],
			"Mfp": float(mfp_mb),
			"Mbp": float(mbp_mb),
		}
		staged_stats.append(stage_cfg)
		sig_to_stats[sig] = stage_cfg

		_stage_clear(stage)
		stage.to("meta")

	# Save results merged into provided config JSON if available
	output_path = args.output
	print(f"\nSaving results to {output_path}")

	merged = None
	base_cfg_path = getattr(args, "config_file", None)
	if base_cfg_path:
		try:
			with open(base_cfg_path, "r") as f:
				merged = json.load(f)
		except Exception:
			merged = None

	if merged is None:
		merged = {"model": config}

	merged["stages"] = staged_stats
	merged["nparams"] = sum(p.numel() for p in model.parameters())

	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w") as f:
		json.dump(merged, f, indent=2)


if __name__ == "__main__":
	main()
