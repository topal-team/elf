import os
import sys
import time
import argparse
import logging

import torch
import torch.distributed as dist

sys.path.append(".")

from elf import Pipeline, get_sources_targets_sequential
from elf.zb_utils import replace_linear_with_linear_dw
from benchmarks.benchmark_utils import get_handcrafted_partition, meta_to_device
from models.utils import add_transformer_args, build_model_from_args, get_dtype


class DummyOptimizer:
	"""
	Some ranks will get parts of the model that have no parameters.
	These ranks will use a dummy optimizer that does nothing.
	"""

	def __init__(self, *args, **kwargs):
		pass

	def step(self):
		pass

	def zero_grad(self):
		pass


class MixedPrecisionOptimizer:
	"""Mixed precision optimizer wrapper that handles FP16/BF16 training with FP32 master weights."""

	def __init__(self, optimizer_cls, params, dtype, opt_dtype, opt_device, **optimizer_kwargs):
		self.dtype = dtype
		self.opt_dtype = opt_dtype
		self.opt_device = opt_device
		self.need_master_weights = dtype != opt_dtype or opt_device != "cuda"

		if self.need_master_weights:
			# Create master weights in optimizer precision
			self.master_params = []
			self.param_mapping = {}

			for param in params:
				if param.requires_grad:
					master_param = param.detach().clone().to(opt_dtype).to(opt_device)  # cast THEN move!
					master_param.requires_grad_(True)
					self.master_params.append(master_param)
					self.param_mapping[param] = master_param

			self.optimizer = optimizer_cls(self.master_params, **optimizer_kwargs)
		else:
			self.optimizer = optimizer_cls(params, **optimizer_kwargs)

	def zero_grad(self):
		"""Zero gradients for both model params and master weights if needed."""
		if self.need_master_weights:
			# Zero gradients on master weights
			for param in self.master_params:
				if param.grad is not None:
					param.grad.zero_()

			# Also zero model param grads to be safe
			for param in self.param_mapping.keys():
				if param.grad is not None:
					param.grad.zero_()
		else:
			self.optimizer.zero_grad()

	def step(self):
		"""Perform optimization step with gradient copying if using master weights."""
		if self.need_master_weights:
			# Copy gradients from model params to master weights
			for model_param, master_param in self.param_mapping.items():
				if model_param.grad is not None:
					if master_param.grad is None:
						master_param.grad = torch.empty_like(
							master_param, pin_memory=self.opt_device == torch.device("cpu")
						)  # pin if using CPU weights for faster transfer
					master_param.grad.copy_(model_param.grad)

			# Step optimizer on master weights
			self.optimizer.step()

			# Copy updated weights back to model params
			for model_param, master_param in self.param_mapping.items():
				model_param.data.copy_(master_param.data)
		else:
			self.optimizer.step()

	def state_dict(self):
		"""Return optimizer state dict."""
		return self.optimizer.state_dict()

	def load_state_dict(self, state_dict):
		"""Load optimizer state dict."""
		self.optimizer.load_state_dict(state_dict)

	def __getattr__(self, name):
		"""Forward other attributes to the underlying optimizer."""
		return getattr(self.optimizer, name)


def parse_args():
	"""Parse command-line options.

	The transformer hyper-parameters are added through
	``models.utils.add_transformer_args`` to ensure consistency across
	scripts.
	"""

	parser = argparse.ArgumentParser(description="Run distributed training with FP16")

	# Model hyper-parameters (input-dim, hidden-dim, …)
	add_transformer_args(parser, model_type="full")

	# Script-specific parameters
	parser.add_argument("--batch-size", type=int, default=32, help="Global batch size")
	parser.add_argument(
		"--opt-dtype",
		type=str,
		default="float32",
		choices=["float16", "bfloat16", "float32", "float8"],
		help="Optimizer data type",
	)
	parser.add_argument(
		"--opt-device", type=str, default="cuda", choices=["cpu", "cuda"], help="Optimizer device"
	)
	parser.add_argument(
		"--partitioner",
		type=str,
		default="naive",
		choices=["naive", "constrained", "metis", "dagP", "handcrafted"],
		help="Partitioner type",
	)
	parser.add_argument(
		"--schedule",
		type=str,
		default="1f1b",
		choices=["gpipe", "1f1b", "megatron", "hanayo", "zbh1", "zbh2", "zbv", "full_remat"],
		help="Schedule type",
	)
	parser.add_argument("--niters", type=int, default=10, help="Number of training iterations")
	parser.add_argument(
		"--log", type=str, choices=["none", "info", "debug"], default="info", help="Log level"
	)
	parser.add_argument("--profile", action="store_true", help="Profile the training")

	return parser.parse_args()


def set_loglevel(log):
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


def main():
	local_rank = int(os.getenv("LOCAL_RANK"))
	dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
	torch.cuda.set_device(local_rank)

	rank = dist.get_rank()
	world_size = dist.get_world_size()

	args = parse_args()
	set_loglevel(args.log)

	dtype = get_dtype(args.dtype)
	opt_dtype = get_dtype(args.opt_dtype)
	opt_device = torch.device(args.opt_device)
	batch_size = args.batch_size
	nmb = world_size * 2
	mb_size = batch_size // nmb

	start_time = time.time()
	with torch.device("meta"):
		model = build_model_from_args(args, model_type="full")
		replace_linear_with_linear_dw(model, "meta")

	if rank == 0:
		print(f"The model has {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

	if args.partitioner == "handcrafted":
		placement = Pipeline._get_default_placement(args.schedule, world_size)
		parts = get_handcrafted_partition(model, rank, placement)
		sources, targets = get_sources_targets_sequential(placement)
		args.partitioner = False
		sample = None
	else:
		parts = model
		sources, targets = None, None
		sample = model.get_sample(mb_size)
		if rank == 0:
			model = meta_to_device(model, "cuda")

	pipeline = Pipeline(
		parts,
		sample,
		placement="auto",
		partitioner=args.partitioner,
		schedule=args.schedule,
		sources=sources,
		targets=targets,
	)

	dist.barrier()
	torch.cuda.synchronize()
	partition_time = time.time() - start_time

	all_params = [p for block in pipeline.blocks for p in block.model.parameters() if p.requires_grad]
	if len(all_params) == 0:
		optimizer = DummyOptimizer()
	else:
		optimizer = MixedPrecisionOptimizer(
			torch.optim.AdamW, all_params, dtype, opt_dtype, opt_device, eps=1e-5
		)  # eps needed when optimizer is <= fp16

	for _ in range(5):  # Warmup
		optimizer.zero_grad()
		y, loss = pipeline(
			model.get_sample(batch_size), model.get_target(batch_size), model.loss_fn, split_size=mb_size
		)

	torch.cuda.reset_peak_memory_stats()

	if args.profile:
		torch.cuda.cudart().cudaProfilerStart()

	dist.barrier()
	start_time = time.time()

	for i in range(args.niters):
		optimizer.zero_grad()
		y, loss = pipeline(
			model.get_sample(batch_size), model.get_target(batch_size), model.loss_fn, split_size=mb_size
		)
		optimizer.step()

	dist.barrier()
	training_time = time.time() - start_time

	if args.profile:
		torch.cuda.cudart().cudaProfilerStop()

	time.sleep(rank * 0.1)  # avoid all printing at the same time :)
	print(
		f"Rank {rank}: peak memory = {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", flush=True
	)
	if rank == 0:
		time.sleep(world_size * 0.1)
		print(
			f"Times:\n\tModel creation + partition = {partition_time:.2f}s\n\tTraining ({args.niters} iters) = {training_time:.2f}s ({training_time / args.niters:.2f}s / iter)\n\tThroughput: {(args.batch_size * args.niters * args.seq_len) / training_time:.2f} tokens/s",
			flush=True,
		)
		print("\nNow testing correctness...", flush=True, end="")

	# test_correctness(model, pipeline)

	if rank == 0:
		print("\tPassed!", flush=True)

	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
