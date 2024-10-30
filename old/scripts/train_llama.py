import os
import sys

sys.path.append("./")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets
from transformers import AutoTokenizer

# from models.llama3 import Llama, ModelArgs
from models.GPT import GPT, GPT13BConfig, GPTLargeConfig
from pipeline import Pipeline

import argparse
import logging

logger = logging.getLogger("train_llama")
logging.basicConfig(level=logging.INFO)


def pretty_print_params(n):
	if n > 1e9:
		return f"{n/1e9:.1f}B"
	elif n > 1e6:
		return f"{n/1e6:.1f}M"
	else:
		return f"{int(n)}"


def parse_args():
	parser = argparse.ArgumentParser(description="Train Llama3 model")
	parser.add_argument(
		"--batch_size", "-bs", type=int, default=32, required=False, help="Batch size for training"
	)
	parser.add_argument(
		"--epochs", "-e", type=int, default=3, required=False, help="Number of epochs to train"
	)
	parser.add_argument("--lr", type=float, default=1e-4, required=False, help="Learning rate")
	parser.add_argument(
		"--max_seq_len", type=int, default=512, required=False, help="Maximum sequence length"
	)
	parser.add_argument(
		"--dataset_path", "-d", type=str, default="/data", required=False, help="Path to dataset"
	)
	parser.add_argument(
		"--tokenizer_path", "-tk", type=str, default="/data", required=False, help="Path to tokenizer"
	)
	parser.add_argument(
		"-dp", type=int, default=1, required=False, help="Number of data parallel processes"
	)
	parser.add_argument(
		"-pp", type=int, default=4, required=False, help="Number of pipeline parallel processes"
	)
	parser.add_argument(
		"--log", choices=["debug", "info", "none"], default="info", required=False, help="logging level"
	)
	args = parser.parse_arg()
	match args.log:
		case "debug":
			logging.getLogger().setLevel(logging.DEBUG)
		case "info":
			logging.getLogger().setLevel(logging.INFO)
		case "none":
			logging.getLogger().setLevel(100)
	return args


def main():
	args = parse_args()

	# Load dataset
	# dataset = datasets.load_from_disk("/data/wikitext-2-v1/train")
	# Initialize tokenizer
	# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
	tokenizer = AutoTokenizer.from_pretrained(f"{args.tokenizer_path}")

	tokenizer.pad_token = tokenizer.eos_token

	if os.path.exists(args.dataset_path + "/tokenized/train"):
		tokenized_dataset = datasets.load_from_disk(args.dataset_path + "/tokenized/train")
	else:
		print(f"No tokenized dataset found at {args.dataset_path}/tokenized/train.")
		exit(1)

	if rank == 0:
		print(f"Loaded {pretty_print_params(len(tokenized_dataset))} samples")
		print(f"Vocab size: {tokenizer.vocab_size}")

	# Create DataLoader
	dataloader = DataLoader(
		tokenized_dataset,
		batch_size=args.batch_size // args.dp,
		sampler=DistributedSampler(tokenized_dataset, num_replicas=args.dp, rank=rank // args.pp),
	)

	# Initialize model
	# model_args = ModelArgs(
	# 	dim=128,
	# 	n_layers=64,
	# 	n_heads=4,
	# 	vocab_size=tokenizer.vocab_size + 2,
	# 	max_seq_len=args.max_seq_len,
	# )

	sample = torch.randint(0, 10, (args.batch_size // args.pp, args.max_seq_len))
	model = GPT(GPTLargeConfig(tokenizer.vocab_size + 2, args.max_seq_len))
	# model = Llama(model_args)
	if rank == 0:
		print(
			"# of trainable parameters : ",
			pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
		)
	placement = list(range(args.pp)) * 2
	pipe = Pipeline(
		model, sample, placement, partition="metis", schedule="afab", dp=args.dp
	)

	# Initialize optimizer
	optimizer = torch.optim.AdamW(pipe.parameters(), lr=args.lr)

	torch.cuda.cudart().cudaProfilerStart()

	def loss_fn(logits, targets):
		logits = logits.view(-1, logits.size(-1))
		targets = targets.view(-1)
		return nn.functional.cross_entropy(logits, targets)

	# Training loop
	model.train()
	for epoch in range(args.epochs):
		total_loss = 0
		i = 0
		for i, batch in enumerate(dataloader):
			# Transform batch["input_ids"] into a single tensor
			input_ids = torch.stack(batch["input_ids"], -1).cuda()

			optimizer.zero_grad()

			_, loss = pipe(
				input_ids, input_ids, loss_fn, split_size=args.batch_size // args.pp, profile=True
			)

			optimizer.step()

			if rank == ws - 1:
				total_loss += loss.item()
				if i % 100 == 99:
					print(f"[{epoch + 1} | {i}] : {total_loss / i:.4f}")

		if rank == ws - 1:
			avg_loss = total_loss / len(dataloader)
			print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

	torch.cuda.cudart().cudaProfilerStop()
	pipe.clear()


if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	ws = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	main()

	if dist.is_initialized():
		dist.destroy_process_group()
