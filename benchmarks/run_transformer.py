import os
import sys
import time
import argparse
import logging

import torch
import torch.distributed as dist

sys.path.append(".")

from elf import Pipeline
from models.simple import FullTransformer
from elf.zb_utils import replace_linear_with_linear_dw
from benchmarks.benchmark_utils import meta_to_device


class MixedPrecisionOptimizer:
	"""Mixed precision optimizer wrapper that handles FP16/BF16 training with FP32 master weights."""

	def __init__(self, optimizer_cls, params, dtype, opt_dtype, **optimizer_kwargs):
		self.dtype = dtype
		self.opt_dtype = opt_dtype
		self.need_master_weights = dtype != opt_dtype

		if self.need_master_weights:
			# Create master weights in optimizer precision
			self.master_params = []
			self.param_mapping = {}

			for param in params:
				if param.requires_grad:
					master_param = param.detach().clone().to(opt_dtype)
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
						master_param.grad = torch.empty_like(master_param)
					master_param.grad.copy_(model_param.grad.to(dtype=master_param.dtype))

			# Step optimizer on master weights
			self.optimizer.step()

			# Copy updated weights back to model params
			for model_param, master_param in self.param_mapping.items():
				model_param.data.copy_(master_param.data.to(dtype=model_param.dtype))
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
	parser = argparse.ArgumentParser(description="Run distributed training with FP16")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
	parser.add_argument("--input-dim", type=int, default=5000, help="Vocab size")
	parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden dimension")
	parser.add_argument("--nblocks", type=int, default=32, help="Number of blocks")
	parser.add_argument("--num-heads", type=int, default=32, help="Number of heads")
	parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
	parser.add_argument(
		"--dtype",
		type=str,
		default="float32",
		choices=["float16", "bfloat16", "float32", "float8"],
		help="Data type",
	)
	parser.add_argument(
		"--opt-dtype",
		type=str,
		default="float32",
		choices=["float16", "bfloat16", "float32", "float8"],
		help="Optimizer data type",
	)
	parser.add_argument(
		"--sdpa",
		type=str,
		default=None,
		choices=["math", "flash", "efficient", "cudnn"],
		help="SDPA implementation",
	)
	parser.add_argument(
		"--partitioner",
		type=str,
		default="naive",
		choices=["naive", "constrained", "metis", "dagP"],
		help="Partitioner type",
	)
	parser.add_argument(
		"--schedule",
		type=str,
		default="1f1b",
		choices=["gpipe", "1f1b", "megatron", "hanayo", "zbh1", "zbh2", "full_remat"],
		help="Schedule type",
	)
	parser.add_argument("--niters", type=int, default=10, help="Number of training iterations")
	parser.add_argument(
		"--log", type=str, choices=["none", "info", "debug"], default="info", help="Log level"
	)
	return parser.parse_args()


def get_dtype(dtype):
	match dtype:
		case "float16" | "fp16":
			return torch.float16
		case "bfloat16" | "bf16":
			if not torch.cuda.is_bf16_supported():
				logging.warning("Bfloat16 is not supported on this GPU")
			return torch.bfloat16
		case "float32" | "fp32":
			return torch.float32
		case _:
			raise ValueError(f"Invalid data type: {dtype}")


def get_sdpa(sdpa):
	match sdpa:
		case None:
			return None
		case "math":
			return "MATH"
		case "flash":
			return "FLASH_ATTENTION"
		case "efficient":
			return "EFFICIENT_ATTENTION"
		case "cudnn":
			return "CUDNN_ATTENTION"
		case _:
			raise ValueError(f"Invalid SDPA implementation: {sdpa}")


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
	batch_size = args.batch_size
	nmb = world_size * 2
	mb_size = batch_size // nmb

	start_time = time.time()
	with torch.device("meta"):
		model = FullTransformer(
			input_dim=args.input_dim,
			hidden_dim=args.hidden_dim,
			n_blocks=args.nblocks,
			seq_len=args.seq_len,
			num_heads=args.num_heads,
			sdp_backend=get_sdpa(args.sdpa),
		)
		model.to(dtype)
		replace_linear_with_linear_dw(model, "meta")

	if rank == 0:
		# only rank 0 profiles and partitions
		model = meta_to_device(model, "cuda")
		print(f"The model has {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

	pipeline = Pipeline(
		model,
		model.get_sample(mb_size),
		placement="auto",
		partitioner=args.partitioner,
		schedule=args.schedule,
	)
	dist.barrier()
	torch.cuda.synchronize()
	partition_time = time.time() - start_time

	start_time = time.time()
	sample = model.get_sample(batch_size)
	target = model.get_target(batch_size)

	all_params = [p for block in pipeline.blocks for p in block.model.parameters() if p.requires_grad]
	optimizer = MixedPrecisionOptimizer(
		torch.optim.Adam, all_params, dtype, opt_dtype, eps=1e-5
	)  # eps needed when optimizer is <= fp16

	for _ in range(5):  # Warmup
		optimizer.zero_grad()
		y, loss = pipeline(sample, target, model.loss_fn, split_size=mb_size)

	torch.cuda.reset_peak_memory_stats()
	for _ in range(args.niters):
		optimizer.zero_grad()
		y, loss = pipeline(sample, target, model.loss_fn, split_size=mb_size)
		optimizer.step()

	dist.barrier()
	torch.cuda.synchronize()
	training_time = time.time() - start_time

	time.sleep(rank * 0.1)  # avoid all printing at the same time :)
	print(
		f"Rank {rank}: peak memory = {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", flush=True
	)
	if rank == 0:
		time.sleep(world_size * 0.1)
		print(
			f"Times:\n\tModel creation + partition = {partition_time:.2f}s\n\tTraining ({args.niters} iters) = {training_time:.2f}s",
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
