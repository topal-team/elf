import time
import json
import argparse
import logging

import torch
import torch.distributed as dist

from elf import (
	Pipeline,
	PipelineConfig,
	get_sources_targets_sequential,
	Placement,
	replace_linear_with_linear_dw,  # noqa: F401
)
from elf.registry import SCHEDULERS
from benchmarks.benchmark_utils import (
	get_checkpointed_scheduler,
	get_offloaded_scheduler,
	get_handcrafted_partition,
	meta_to_device,
	init_dist,
)
from models.utils import add_transformer_args, build_model_from_args, get_dtype

logger = logging.getLogger("run_transformer")


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
		self.need_master_weights = dtype != opt_dtype or opt_device != torch.device("cuda")

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


def write_detailed_stats(detailed_stats, stats_file):
	rank, world_size = dist.get_rank(), dist.get_world_size()
	for r in range(world_size):
		dist.barrier()
		if r == rank:
			if r == 0:
				with open(stats_file, "w") as f:
					json.dump({rank: detailed_stats}, f, indent=2)
			else:
				with open(stats_file, "r") as f:
					stats_data = json.load(f)
				stats_data[rank] = detailed_stats
				with open(stats_file, "w") as f:
					json.dump(stats_data, f, indent=2)


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
	parser.add_argument(
		"--batch-size", type=int, default=None, help="Global batch size, per DP group"
	)
	parser.add_argument("--nmb", type=int, default=None, help="Number of micro-batches")
	parser.add_argument("--mb-size", type=int, default=1, help="Micro-batch size")
	parser.add_argument(
		"--opt-dtype",
		type=str,
		default=None,
		choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
		help="Optimizer data type (defaults to same as dtype)",
	)
	parser.add_argument(
		"--opt-device", type=str, default="cuda", choices=["cpu", "cuda"], help="Optimizer device"
	)
	parser.add_argument("--no-optimizer", action="store_true", help="Do not use optimizer")
	parser.add_argument(
		"--partitioner",
		type=str,
		default="naive",
		choices=["naive", "constrained", "metis", "dagP", "handcrafted"],
		help="Partitioner type",
	)
	parser.add_argument(
		"--scheduler",
		type=str,
		default="1f1b",
		choices=["gpipe", "1f1b", "megatron", "hanayo", "zbh1", "zbh2", "zbv", "full_remat", "file"],
		help="Schedule type",
	)
	parser.add_argument("--schedule-file", type=str, help="Schedule from file")
	parser.add_argument("--interleaving", type=int, default=1, help="Interleaving factor")
	parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")
	parser.add_argument("--niters", type=int, default=10, help="Number of training iterations")
	parser.add_argument(
		"--checkpointing",
		type=str,
		default=None,
		choices=["full", "simple", "selective"],
		help="Checkpointing strategy",
	)
	parser.add_argument("--offloading", type=int, default=None, help="Offloading ratio")
	parser.add_argument("--prefetching-time", type=int, default=1, help="Prefetching time")
	parser.add_argument(
		"--log", type=str, choices=["none", "info", "debug"], default="info", help="Log level"
	)
	parser.add_argument(
		"--backend", type=str, default="nccl", choices=["nccl", "mpi"], help="Distributed backend"
	)
	parser.add_argument("--stats-file", type=str, help="File to save detailed stats to")
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
	args = parse_args()

	local_rank, rank, world_size = init_dist(args.backend)

	set_loglevel(args.log)

	pp = world_size // args.dp

	mb_size = args.mb_size
	assert not (args.batch_size is None and args.nmb is None), (
		"Batch size or number of micro-batches must be provided"
	)
	assert (args.batch_size is None or args.nmb is None) or (
		args.nmb * args.mb_size == args.batch_size
	), "If both batch size and number of micro-batches are provided, they must be consistent"
	batch_size = mb_size * args.nmb if args.batch_size is None else args.batch_size
	assert batch_size % mb_size == 0, "Batch size must be divisible by micro-batch size"

	if args.scheduler == "file":
		assert args.schedule_file is not None, "Schedule file is required for file scheduler"
		with open(args.schedule_file, "r") as f:
			schedule_dict = json.load(f)
		scheduler = SCHEDULERS["fixed"](schedule_dict)
	else:
		scheduler = args.scheduler

	placement = (
		Placement.default(args.scheduler, pp) * args.interleaving
	)  # get placement before potentially changing the scheduler

	if args.checkpointing is not None:
		scheduler = get_checkpointed_scheduler(scheduler, args.checkpointing)

	elif args.offloading is not None:
		scheduler = get_offloaded_scheduler(scheduler, args.offloading, args.prefetching_time)

	start_time = time.time()
	with torch.device("meta"):
		model, dtype = build_model_from_args(args, model_type="full")
		# replace_linear_with_linear_dw(model, "meta")

	if rank == 0:
		print(f"The model has {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

	if args.partitioner == "handcrafted":
		parts = get_handcrafted_partition(model, rank, placement)
		sources, targets = get_sources_targets_sequential(placement)
		args.partitioner = False
		sample = None
	else:
		parts = model
		sources, targets = None, None
		sample = model.get_sample(mb_size, dtype=dtype)
		if rank == 0:
			model = meta_to_device(model, "cuda")

	pipe_config = PipelineConfig(
		partitioner=args.partitioner, scheduler=scheduler, placement=placement, pp=pp, dp=args.dp
	)

	pipeline = Pipeline(parts, sample, config=pipe_config, sources=sources, targets=targets)

	dist.barrier()
	torch.cuda.synchronize()
	partition_time = time.time() - start_time

	opt_dtype = get_dtype(args.opt_dtype) if args.opt_dtype is not None else dtype
	opt_device = torch.device(args.opt_device)

	all_params = [p for block in pipeline.blocks for p in block.model.parameters() if p.requires_grad]
	if len(all_params) == 0 or args.no_optimizer:
		optimizer = DummyOptimizer()
	else:
		optimizer = MixedPrecisionOptimizer(
			torch.optim.AdamW, all_params, dtype, opt_dtype, opt_device, eps=1e-5
		)  # eps needed when optimizer is <= fp16

	for _ in range(5):  # Warmup
		optimizer.zero_grad()
		y, loss = pipeline(
			model.get_sample(batch_size, dtype=dtype),
			model.get_target(batch_size, dtype=dtype),
			model.loss_fn,
			split_size=mb_size,
		)

	data = [
		(
			model.get_sample(batch_size, dtype=dtype, device="cuda"),
			model.get_target(batch_size, dtype=dtype, device="cuda"),
		)
		for _ in range(args.niters)
	]

	torch.cuda.reset_peak_memory_stats()
	torch.cuda.synchronize()
	dist.barrier()

	if args.profile:
		torch.cuda.cudart().cudaProfilerStart()

	start_time = time.time()

	for sample, target in data:
		optimizer.zero_grad()
		y, loss = pipeline(sample, target, model.loss_fn, split_size=mb_size, profile=args.profile)
		optimizer.step()

	torch.cuda.synchronize()
	dist.barrier()
	training_time = time.time() - start_time

	if args.profile:
		torch.cuda.cudart().cudaProfilerStop()

	time.sleep(rank * 0.1)  # avoid all printing at the same time :)
	if args.stats_file is not None:
		write_detailed_stats(pipeline.detailed_stats, args.stats_file)
	else:
		print(
			f"Rank {rank}: peak memory = {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", flush=True
		)

	if rank == 0:
		time.sleep(world_size * 0.1)
		print(
			f"Times:\n\tModel creation + partition = {partition_time:.2f}s\n\tTraining ({args.niters} iters) = {training_time:.2f}s ({training_time / args.niters:.2f}s / iter)\n\tThroughput: {(args.dp * batch_size * args.niters * model.seq_len) / training_time:.2f} tokens/s",
			flush=True,
		)

		# print("\nNow testing correctness...", flush=True, end="")

	# test_correctness(model, pipeline)
	# if rank == 0:
	# print("\tPassed!", flush=True)

	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
