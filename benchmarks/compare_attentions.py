import os
import sys
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.profiler as profiler

from torch.cuda import cudart

sys.path.append(".")

from elf import Pipeline
from elf.utils import TimerGPU
from elf.pipeline import get_sources_targets_sequential
from models.simple import FullTransformer, ChainTransformer
from models.utils import get_dtype, get_sdpa
from benchmark_utils import meta_to_device

import argparse


def get_blocks(rank, placements, model):
	return [b for (i, b) in zip(placements, model.blocks) if i == rank]


def get_grouped_blocks(rank, placement):
	num_ranks = len(placement)
	parts = [None] * num_ranks

	blocks_per_stage = len(model.blocks) // len(placement)
	start_idx = 0
	for i in range(num_ranks):
		end_idx = start_idx + blocks_per_stage

		if isinstance(model, FullTransformer) and i == 0:
			parts[i] = torch.nn.Sequential(model.embed, *model.blocks[start_idx:end_idx])
		else:
			parts[i] = torch.nn.Sequential(*model.blocks[start_idx:end_idx])

		if isinstance(model, FullTransformer) and i == num_ranks - 1:
			parts[i].append(model.head)  # doesn't work for multi waves

		start_idx = end_idx

	parts = [parts[i] for i, p in enumerate(placement) if p == rank]
	parts = [meta_to_device(p) for p in parts]
	return parts


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"]
	)
	parser.add_argument(
		"--sdp-backend", type=str, default=None, choices=[None, "FLASH_ATTENTION", "MATH"]
	)
	parser.add_argument("--transformer-type", type=str, default="chain", choices=["full", "chain"])
	return parser.parse_args()


if __name__ == "__main__":
	rank = int(os.environ.get("RANK", 0))
	local_rank = int(os.environ.get("LOCAL_RANK", 0))
	ws = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=240))

	args = parse_args()

	nmb = ws * 2
	mb_size = 8
	batch_size = nmb * mb_size

	num_heads = 32
	seq_len = 1024
	head_dim = 128
	n_blocks = 8
	embed_dim = num_heads * head_dim
	hidden_dim = embed_dim  # 4096
	placements = [0, 1, 2, 3, 4, 5, 6, 7][:4]

	dtype = get_dtype(args.dtype)
	sdp_backend = get_sdpa(args.sdp_backend)

	if torch.cuda.is_available():
		# device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
		device = local_rank
	else:
		device = torch.device("cpu")

	if args.transformer_type == "full":
		with torch.device("meta"):
			model = FullTransformer(
				input_dim=embed_dim,
				hidden_dim=hidden_dim,
				n_blocks=n_blocks,
				seq_len=seq_len,
				num_heads=num_heads,
				sdp_backend=sdp_backend,
			).to(dtype)
	else:
		with torch.device("meta"):
			model = ChainTransformer(
				hidden_dim=hidden_dim,
				n_blocks=n_blocks,
				seq_len=seq_len,
				num_heads=num_heads,
				sdp_backend=sdp_backend,
			).to(dtype)

	sample = model.get_sample(batch_size).to(device)  # keep as long for embedding
	if args.transformer_type == "chain":
		sample = sample.to(dtype)

	target = model.get_target(batch_size).to(device)
	if args.transformer_type == "chain":
		target = target.to(dtype)

	rank_blocks = get_grouped_blocks(rank, placements)
	sources, targets = get_sources_targets_sequential(placements)

	pipe = Pipeline(
		rank_blocks,
		sample,
		scheduler="1f1b",
		partitioner=False,
		placement=placements,
		sources=sources,
		targets=targets,
	)
	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(pipe.parameters())

	# Warmup iterations
	for _ in range(3):
		_ = pipe(sample, target, loss_fn, split_size=mb_size, profile=False)
		optimizer.step()

	torch.cuda.synchronize()

	# Profile iterations
	profiler.start()
	cudart().cudaProfilerStart()

	with TimerGPU() as timer:
		for i in range(10):
			if rank == 0:
				print(f"Iteration {i}")
			_ = pipe(sample, target, loss_fn, split_size=mb_size, profile=False)
			optimizer.step()

	print(f"Rank: {rank}, time: {timer.time()}")
	cudart().cudaProfilerStop()
	profiler.stop()

	pipe.clear()

	if dist.is_initialized():
		# dist.barrier()
		dist.destroy_process_group()
