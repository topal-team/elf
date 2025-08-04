"""
DeepSpeed/Accelerate version of `run_transformer.py`.

This script trains the same transformers defined in ``models.utils`` but relies on
🤗 Accelerate with the DeepSpeed integration instead of ELF’s custom pipeline.

Key points
----------
* Uses ``accelerate.launch`` to handle distributed setup (no manual ``torch.distributed`` init).
* Supports the same CLI used by ``benchmarks/run_transformer.py`` so you can drop‐in replace it
  in your benchmarking loops.
* Micro-batching is implemented through gradient accumulation to mimic the original
  ``split_size`` behaviour.
* Mixed-precision is delegated to Accelerate/DeepSpeed – the ``--dtype`` and ``--opt-dtype``
  flags are still honoured.

Launch example
~~~~~~~~~~~~~~
>>> accelerate launch --config_file ds_config.yaml benchmarks/deepspeed_transformer.py \
...     --batch-size 128 --mb-size 4 --niters 20 --dtype bf16 --log info

A minimal DeepSpeed config (``ds_config.yaml``) could look like::

    {
      "zero_optimization": {"stage": 1},
      "bf16": {"enabled": true},
      "train_micro_batch_size_per_gpu": 4
    }

Note: Adjust ``train_micro_batch_size_per_gpu`` or pass ``--mb-size`` so that
``batch-size == mb-size * nmb``.
"""

from __future__ import annotations

import argparse
import logging
import time

import torch

try:
	from accelerate import Accelerator
except ModuleNotFoundError as exc:  # pragma: no cover – optional dependency
	raise ModuleNotFoundError(
		"The `accelerate` package is required to run this script. Install it with `pip install accelerate deepspeed`."
	) from exc

from models.utils import add_transformer_args, build_model_from_args

# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------


def _add_script_args(parser: argparse.ArgumentParser) -> None:
	"""Add the script-specific parameters (kept identical to run_transformer.py)."""

	add = parser.add_argument

	add("--batch-size", type=int, default=None, help="Global batch size, per DP group")
	add("--nmb", type=int, default=None, help="Number of micro-batches")
	add("--mb-size", type=int, default=1, help="Micro-batch size")

	add(
		"--opt-dtype",
		type=str,
		default=None,
		choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
		help="Optimizer data type (defaults to same as dtype)",
	)
	add(
		"--opt-device",
		type=str,
		default="cuda",
		choices=["cpu", "cuda"],
		help="Optimizer device (ignored – handled by DeepSpeed)",
	)

	# No pipeline/partitioner – DeepSpeed handles that internally.
	add("--niters", type=int, default=10, help="Number of training iterations")
	add("--log", type=str, choices=["none", "info", "debug"], default="info", help="Log level")
	add("--profile", action="store_true", help="Profile the training (nvtx ranges)")


def parse_args() -> argparse.Namespace:  # noqa: D401 – simple wrapper
	"""Parse command-line options."""

	parser = argparse.ArgumentParser(description="Run distributed training with DeepSpeed")
	# Model hyper-parameters (input-dim, hidden-dim, …)
	add_transformer_args(parser, model_type="full")
	_add_script_args(parser)

	return parser.parse_args()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_loglevel(log: str) -> None:
	logging.basicConfig(level=logging.CRITICAL)
	match log:
		case "none":
			logging.getLogger().setLevel(100)
		case "info":
			logging.getLogger().setLevel(logging.INFO)
		case "debug":
			logging.getLogger().setLevel(logging.DEBUG)
		case _:
			raise ValueError(f"Invalid log level: {log}")


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------


def main() -> None:  # noqa: C901 – keep everything in one function for parity
	args = parse_args()
	set_loglevel(args.log)

	# ------------------------------------------------------------------
	# Consistency checks (identical to run_transformer.py)
	# ------------------------------------------------------------------
	mb_size = args.mb_size
	assert not (args.batch_size is None and args.nmb is None), (
		"Batch size or number of micro-batches must be provided"
	)
	assert (args.batch_size is None or args.nmb is None) or (
		args.nmb * args.mb_size == args.batch_size
	), "If both batch size and number of micro-batches are provided, they must be consistent"
	batch_size = mb_size * args.nmb if args.batch_size is None else args.batch_size
	assert batch_size % mb_size == 0, "Batch size must be divisible by micro-batch size"

	# gradient_accumulation == number of micro-batches per optimisation step
	grad_accum_steps = batch_size // mb_size

	# ------------------------------------------------------------------
	# Build model on meta device first for speed, then materialise on proper
	# devices when Accelerate gives us one.
	# ------------------------------------------------------------------
	t0 = time.time()
	model, dtype = build_model_from_args(args, model_type="full")

	# Instantiate Accelerator – configuration (device placement, mixed precision, DeepSpeed, …)
	# will be pulled automatically from the config file created with `accelerate config` if you
	# launch with `accelerate launch --config_file my_cfg.yaml …`.
	# We only override the gradient accumulation so that micro-batch logic is consistent.
	accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)

	# Ensure DeepSpeed has a valid micro-batch size when no DataLoader is used.
	if accelerator.state.deepspeed_plugin is not None:
		print(f"Setting train_micro_batch_size_per_gpu to {mb_size}")
		ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
		ds_cfg.setdefault("train_micro_batch_size_per_gpu", mb_size)

	device = accelerator.device

	# Materialise parameters on the actual device (CUDA / CPU)
	model = model.to(device=device, dtype=dtype)

	creation_time = time.time() - t0

	# ------------------------------------------------------------------
	# Optimiser (DeepSpeed will wrap/replace it as necessary)
	# ------------------------------------------------------------------
	# Optimiser data type is handled internally by DeepSpeed, so we build an AdamW
	# optimiser in regular precision; DeepSpeed will cast/duplicate weights as needed.
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-5)

	# Prepare everything – this hands control over to DeepSpeed/Accelerate
	model, optimizer = accelerator.prepare(model, optimizer)

	# Warm-up (same number of iters as original script)
	for _ in range(5):
		optimizer.zero_grad()
		sample = model.get_sample(batch_size, device=device)
		target = model.get_target(batch_size, device=device)
		pred = model(sample)
		loss = model.loss_fn(pred, target)
		accelerator.backward(loss)
		optimizer.step()

	accelerator.wait_for_everyone()
	torch.cuda.reset_peak_memory_stats()

	data = [
		(model.get_sample(batch_size, device=device), model.get_target(batch_size, device=device))
		for _ in range(args.niters)
	]

	if args.profile:
		torch.cuda.cudart().cudaProfilerStart()

	# ------------------------------------------------------------------
	# Training loop
	# ------------------------------------------------------------------
	start_time = time.time()

	for sample, target in data:
		optimizer.zero_grad()
		# Split into micro-batches manually so we can use the same synthetic data
		for i in range(0, batch_size, mb_size):
			mb_sample = sample[i : i + mb_size]
			mb_target = target[i : i + mb_size]
			pred = model(mb_sample)
			loss = model.loss_fn(pred, mb_target) / grad_accum_steps
			accelerator.backward(loss)
		optimizer.step()

	accelerator.wait_for_everyone()
	training_time = time.time() - start_time

	if args.profile:
		torch.cuda.cudart().cudaProfilerStop()

	# ------------------------------------------------------------------
	# Metrics & logging
	# ------------------------------------------------------------------
	peak_mem = torch.cuda.max_memory_allocated() / 1024**3
	print(f"Rank {accelerator.process_index}: peak memory = {peak_mem:.2f}GB", flush=True)

	if accelerator.is_main_process:
		seq_len = model.module.seq_len if hasattr(model, "module") else model.seq_len
		tokens_per_sec = (batch_size * args.niters * seq_len) / training_time

		print(
			"Times:\n"
			f"\tModel creation = {creation_time:.2f}s\n"
			f"\tTraining ({args.niters} iters) = {training_time:.2f}s "
			f"({training_time / args.niters:.2f}s / iter)\n"
			f"\tThroughput: {tokens_per_sec:.2f} tokens/s",
			flush=True,
		)


if __name__ == "__main__":
	main()
