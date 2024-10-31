import os
import sys
import time
from datetime import timedelta

sys.path.append("./")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets
from transformers import AutoTokenizer

from models.GPT import GPT, GPTXXXLConfig
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
		"--tokenizer",
		type=str,
		default="mistralai/Mistral-7B-v0.1",
		required=False,
		help="Path to tokenizer",
	)
	parser.add_argument(
		"--dataset_size",
		"-ds",
		type=int,
		default=None,
		required=False,
		help="Max size of dataset, in number of sequences",
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
	args = parser.parse_args()
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

	# Initialize tokenizer
	tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
	tokenizer.pad_token = tokenizer.eos_token
	# if rank == 0:
	# 	tokenizer.save_pretrained("/net/storage/pr3/project/tutorial/elf/")
	# 	print("Saved tokenizer")

	# Load dataset
	if os.path.exists(args.dataset_path + "/tokenized/train"):
		tokenized_dataset = datasets.load_from_disk(args.dataset_path + "/tokenized/train")
		if args.dataset_size is not None:
			tokenized_dataset = tokenized_dataset.select(range(args.dataset_size))
	else:
		print(f"No tokenized dataset found at {args.dataset_path}/tokenized/train.")
		exit(1)

	if rank == 0:
		print(f"Loaded {pretty_print_params(len(tokenized_dataset))} samples")
		print(f"Vocab size: {tokenizer.vocab_size}")

	sampler = DistributedSampler(tokenized_dataset, num_replicas=args.dp, rank=rank // args.pp)
	# Create DataLoader
	dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, sampler=sampler)

	# Initialize model
	model_args = GPTXXXLConfig(tokenizer.vocab_size + 2, args.max_seq_len)
	model = GPT(model_args)

	if rank == 0:
		print(
			"# of trainable parameters : ",
			pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
		)
	# placement = list(range(args.pp)) + list(reversed(range(args.pp)))
	placement = list(range(args.pp)) * 2
	sample = torch.randint(0, 10, (args.batch_size // len(placement), args.max_seq_len))
	pipe = Pipeline(
		model, sample, placement, partition="metis", schedule="1f1b", dp=args.dp, worker=1
	)

	# Initialize optimizer
	optimizer = torch.optim.AdamW(pipe.parameters(), lr=args.lr)

	def loss_fn(logits, targets, *args, **kwargs):
		logits = logits.view(-1, logits.size(-1))
		targets = torch.roll(targets, shifts=-1)
		targets[:, -1] = -100
		targets = targets.view(-1)
		return nn.functional.cross_entropy(logits, targets, ignore_index=-100, *args, **kwargs)

	model.train()

	start = time.time()
	torch.cuda.cudart().cudaProfilerStart()

	# Training loop
	for epoch in range(args.epochs):
		sampler.set_epoch(epoch)
		total_loss = 0
		i = 0
		start = time.time()
		for i, batch in enumerate(dataloader):
			# Transform batch["input_ids"] into a single tensor
			input_ids = torch.stack(batch["input_ids"], -1).cuda()

			optimizer.zero_grad()

			_, loss = pipe(
				input_ids, input_ids, loss_fn, split_size=args.batch_size // len(placement), profile=True
			)
			if loss:
				total_loss += loss.item()

			optimizer.step()

			if i % 50 == 49:
				if rank == ws - 1:
					print(f"[{epoch + 1} | {i}] : {total_loss / (i + 1):.4f}")
				if rank < args.pp:
					print(
						f"Rank {rank} - Last registered times : {pipe.times}s | Peak memory : {torch.cuda.max_memory_allocated() / (2 ** 30):.2f}GB"
					)
					torch.cuda.reset_peak_memory_stats()

		if rank == ws - 1:
			avg_loss = total_loss / len(dataloader)
			print(
				f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f} (Time taken: {time.time() - start:.4f}s)"
			)
		pipe.save(
			f"/net/scratch/hscra/project/tutorial/tutorial051/checkpoint_{epoch + 1}.pt", worker=1
		)

	torch.cuda.synchronize()
	end = time.time()
	if rank == 0:
		print(f"Total time: {end - start:.4f}s")
	torch.cuda.cudart().cudaProfilerStop()
	pipe.clear()


if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	ws = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

	main()

	if dist.is_initialized():
		dist.destroy_process_group()
