import os
import sys
import argparse

import torch
import torch.distributed as dist

sys.path.append(".")

from models.llama3 import Llama, ModelArgs
from elf import Pipeline, get_sources_targets_sequential
from elf.utils import Timer
from elf.zb_utils import replace_linear_with_linear_dw
from benchmarks.benchmark_utils import meta_to_device


def get_parts_model(model, rank, placement):
	"""
	Partition the model into parts for each rank.
	"""
	num_blocks = len(model.layers)
	blocks_per_rank = num_blocks // len(placement)
	num_ranks = len(placement)
	parts = [None] * num_ranks

	removed_from_first = 1
	removed_from_last = 3  # the loss computation is very heavy when vocab size is large ; we remove more blocks from the last rank to balance the load

	assert (removed_from_first + removed_from_last) % (len(placement) - 2) == 0, (
		"The unbalanced should be such as the blocks removed from first and last gpu can be distributed equally to the other ranks"
	)

	redistributed_per_rank = (removed_from_first + removed_from_last) // (len(placement) - 2)

	start_idx = 0
	for i in range(num_ranks):
		end_idx = start_idx + blocks_per_rank

		if i == 0:
			end_idx -= removed_from_first
			parts[i] = torch.nn.Sequential(model.tok_embeddings, *model.layers[start_idx:end_idx])
		else:
			end_idx += redistributed_per_rank
			parts[i] = torch.nn.Sequential(*model.layers[start_idx:end_idx])

		if i == num_ranks - 1:
			end_idx -= removed_from_last
			parts[i].append(model.norm)
			parts[i].append(model.output)

		start_idx = end_idx

	parts = [parts[i] for i, p in enumerate(placement) if p == rank]
	parts = [meta_to_device(p, "cuda") for p in parts]
	return parts


def get_config(config_name):
	"""Get model configuration by name."""
	configs = {
		"8b": {"dim": 4096, "ffn_dim": 14336, "n_layers": 32, "n_heads": 32},
		"70b": {"dim": 8192, "ffn_dim": 28672, "n_layers": 80, "n_heads": 64},
		"405b": {"dim": 16384, "ffn_dim": 53248, "n_layers": 126, "n_heads": 128},
	}

	if config_name not in configs:
		raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

	return configs[config_name]


def get_dtype(precision):
	match precision:
		case "fp16":
			return torch.float16
		case "fp32":
			return torch.float32
		case "bf16":
			return torch.bfloat16
		case _:
			raise ValueError(f"Unknown precision: {precision}. Available: fp16, fp32, bf16")


def init_distributed():
	local_rank = int(os.environ.get("LOCAL_RANK", -1))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl", device_id=torch.device(local_rank))

	return dist.get_rank(), dist.get_world_size()


def main():
	rank, world_size = init_distributed()

	parser = argparse.ArgumentParser(description="Llama model benchmark")
	parser.add_argument(
		"--config", choices=["8b", "70b", "405b"], required=True, help="Model configuration to use"
	)
	parser.add_argument("--dim", type=int, help="Override model dimension")
	parser.add_argument("--ffn-dim", type=int, help="Override FFN dimension")
	parser.add_argument("--n-layers", type=int, help="Override number of layers")
	parser.add_argument("--n-heads", type=int, help="Override number of heads")
	parser.add_argument("--vocab-size", type=int, default=128000, help="Vocabulary size")
	parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length")
	parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp16", help="Precision")
	parser.add_argument("--pp", type=int, default=4, help="Pipeline parallelism degree")
	parser.add_argument("--dp", type=int, default=1, help="Data parallelism degree")

	args = parser.parse_args()

	assert args.pp * args.dp == world_size, (
		"Pipeline and data parallelism degrees must match world size"
	)

	# Get base configuration
	config = get_config(args.config)

	# Override with CLI arguments if provided
	if args.dim is not None:
		config["dim"] = args.dim
	if args.ffn_dim is not None:
		config["ffn_dim"] = args.ffn_dim
	if args.n_layers is not None:
		config["n_layers"] = args.n_layers
	if args.n_heads is not None:
		config["n_heads"] = args.n_heads

	dtype = get_dtype(args.dtype)
	batch_size = 2 * args.pp

	# Create model with configuration
	model_args = ModelArgs(
		dim=config["dim"],
		n_layers=config["n_layers"],
		n_heads=config["n_heads"],
		vocab_size=args.vocab_size,
		ffn_dim=config["ffn_dim"],
	)

	with torch.device("meta"):
		model = Llama(model_args).to(dtype)
		replace_linear_with_linear_dw(model, "meta")
		if rank == 0:
			print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

	placement = list(range(args.pp))
	parts = get_parts_model(model, rank, placement)
	sources, targets = get_sources_targets_sequential(placement)

	pipeline = Pipeline(
		parts,
		None,
		placement=placement,
		partitioner=False,
		schedule="zbh1",
		dp=args.dp,
		worker=0,
		sources=sources,
		targets=targets,
	)

	inputs = torch.randint(0, args.vocab_size, (batch_size, args.seq_len))
	targets = torch.randint(0, args.vocab_size, (batch_size, args.seq_len))

	# warmup
	for _ in range(3):
		y, loss = pipeline(inputs, targets, model.loss, split_size=1)

	dist.barrier()

	torch.cuda.cudart().cudaProfilerStart()

	n = 10
	with Timer() as t:
		for i in range(n):
			y, loss = pipeline(inputs, targets, model.loss, split_size=1)

	dist.barrier()

	torch.cuda.cudart().cudaProfilerStop()

	if rank == 0:
		print(
			f"Time per iteration: {t.time() / n:.2f} seconds\nThroughput: {args.seq_len * batch_size * n / t.time():.2f} tokens/s"
		)

	# print(f"Created {args.config} model with:")
	# print(f"  vocab_size: {args.vocab_size}")
	# print(f"  dim: {config['dim']}")
	# print(f"  ffn_dim: {config['ffn_dim']}")
	# print(f"  n_layers: {config['n_layers']}")
	# print(f"  n_heads: {config['n_heads']}")
	# print(f"  seq_len: {args.seq_len}")

	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
