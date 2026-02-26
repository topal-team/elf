import os
import time
import argparse
import logging

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from elf.utils import Timer
from elf import Pipeline, replace_linear_with_linear_dw
from models.simple import FullTransformer

logger = logging.getLogger("demo")
logging.getLogger().setLevel(100)

vocab_size = 16000
hidden_dim = 4096
n_blocks = 16
seq_len = 2048
num_heads = 32
batch_size = 2
sdp_backend = "EFFICIENT_ATTENTION"
dtype = torch.float16
n_iters = 10


def create_model():
	with torch.device("cuda"):
		model = FullTransformer(
			input_dim=vocab_size,
			hidden_dim=hidden_dim,
			n_blocks=n_blocks,
			seq_len=seq_len,
			num_heads=num_heads,
			sdp_backend=sdp_backend,
		).to(dtype)
	return model


def bench_baseline():
	model = create_model()
	optim = torch.optim.SGD(model.parameters())
	print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params transformer")

	# Warmups
	for _ in range(3):
		with torch.device("cuda"):
			sample = model.get_sample(batch_size=batch_size, device="cuda")
			target = model.get_target(batch_size=batch_size, device="cuda")

		output = model(sample)
		loss = model.loss_fn(output, target)
		loss.backward()

	with Timer() as timer:
		for _ in range(n_iters):
			with torch.device("cuda"):
				sample = model.get_sample(batch_size=batch_size, device="cuda")
				target = model.get_target(batch_size=batch_size, device="cuda")

			optim.zero_grad()
			output = model(sample)
			loss = model.loss_fn(output, target)
			loss.backward()
			optim.step()

	print("-- Single GPU --")
	print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
	print(f"Iteration time: {timer.time() / n_iters:.2f} s")
	print(f"Throughput: {n_iters * batch_size / timer.time():.2f} seq / s")


def bench_elf():
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	model = create_model()
	replace_linear_with_linear_dw(model, "cuda")

	if rank == 0:
		print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params transformer")

	sample = model.get_sample(batch_size=1, device="cuda")
	pipe = Pipeline(model, sample, scheduler="zbv")
	optim = torch.optim.SGD(pipe.parameters())

	# Warmups
	for _ in range(3):
		sample = model.get_sample(batch_size=batch_size * world_size, device="cuda")
		target = model.get_target(batch_size=batch_size * world_size, device="cuda")
		output, loss = pipe(sample, target, model.loss_fn, split_size=1)

	with Timer() as timer:
		for _ in range(n_iters):
			optim.zero_grad()
			sample = model.get_sample(batch_size=batch_size * world_size, device="cuda")
			target = model.get_target(batch_size=batch_size * world_size, device="cuda")
			output, loss = pipe(sample, target, model.loss_fn, split_size=1)
			optim.step()

	print(f"Peak memory (rank {rank}): {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
	if rank == 0:
		time.sleep(0.5)
		throughput = n_iters * batch_size * world_size / timer.time()
		print(f"-- {world_size} GPUs (ELF) --")
		print(f"Iteration time: {timer.time() / n_iters:.2f} s")
		print(f"Throughput: {throughput:.2f} seq / s (per gpu: {throughput / world_size:.2f} seq / s)")


def bench_dp():
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	model = create_model()
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
	optim = torch.optim.SGD(model.parameters())

	if rank == 0:
		print(
			f"Model: {sum(p.numel() for p in model.module.parameters()) / 1e9:.2f}B params transformer"
		)

	# Warmups
	for _ in range(3):
		with torch.device("cuda"):
			sample = model.module.get_sample(batch_size=batch_size, device="cuda")
			target = model.module.get_target(batch_size=batch_size, device="cuda")

		output = model(sample)
		loss = model.module.loss_fn(output, target)
		loss.backward()

	with Timer() as timer:
		for _ in range(n_iters):
			optim.zero_grad()
			with torch.device("cuda"):
				sample = model.module.get_sample(batch_size=batch_size, device="cuda")
				target = model.module.get_target(batch_size=batch_size, device="cuda")

			output = model(sample)
			loss = model.module.loss_fn(output, target)
			loss.backward()
			optim.step()

	print(f"Peak memory (rank {rank}): {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
	if rank == 0:
		time.sleep(0.5)
		throughput = n_iters * batch_size * world_size / timer.time()
		print(f"-- {world_size} GPUs (DP) --")
		print(f"Iteration time: {timer.time() / n_iters:.2f} s")
		print(
			f"Throughput: {throughput:.2f} seq / s (per gpu: {throughput / dist.get_world_size():.2f} seq / s)"
		)


def bench_fsdp():
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	model = create_model()
	if rank == 0:
		# FSDP wraps the model, but .parameters() is available as normal
		print(
			f"Model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params transformer (FSDP)"
		)
	# Shard each Transformer block, then the root module
	for block in model.blocks:
		fully_shard(block)
	fully_shard(model)
	optim = torch.optim.SGD(model.parameters())

	with Timer() as timer:
		for _ in range(n_iters):
			with torch.device("cuda"):
				sample = model.get_sample(batch_size=batch_size, device="cuda")
				target = model.get_target(batch_size=batch_size, device="cuda")

			output = model(sample)
			loss = model.loss_fn(output, target)
			loss.backward()
			optim.step()

	print(f"Peak memory (rank {rank}): {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
	if rank == 0:
		time.sleep(0.5)
		throughput = n_iters * batch_size * world_size / timer.time()
		print(f"-- {world_size} GPUs (FSDP) --")
		print(f"Iteration time: {timer.time() / n_iters:.2f} s")
		print(f"Throughput: {throughput:.2f} seq / s (per gpu: {throughput / world_size:.2f} seq / s)")


def init_distributed():
	local_rank = int(os.getenv("LOCAL_RANK", -1))
	if local_rank == -1:
		return
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["base", "elf", "dp", "fsdp"], required=True)
	args = parser.parse_args()

	init_distributed()

	match args.mode:
		case "base":
			bench_baseline()
		case "elf":
			bench_elf()
		case "dp":
			bench_dp()
		case "fsdp":
			bench_fsdp()
		case _:
			raise ValueError(f"Invalid mode: {args.mode}")

	if dist.is_initialized():
		dist.destroy_process_group()
