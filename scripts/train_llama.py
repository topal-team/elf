import sys

sys.path.append("./")
import torch
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
from models.llama3 import Llama, ModelArgs
import argparse
from tqdm import tqdm

import os
import torch.distributed as dist
from pipeline import Pipeline

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
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
	parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
	parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
	parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
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


def tokenize_function(examples, tokenizer, max_length):
	return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)


def main():
	args = parse_args()

	# Load dataset
	dataset = datasets.load_from_disk("/data/wikitext-2-v1/train")

	# Initialize tokenizer
	tokenizer = AutoTokenizer.from_pretrained("/models/meta-llama/Meta-Llama-3.1-8B")
	tokenizer.pad_token = tokenizer.eos_token

	# Tokenize dataset
	tokenized_dataset = dataset.map(
		lambda examples: tokenize_function(examples, tokenizer, args.max_seq_len),
		batched=True,
		remove_columns=dataset.column_names,
	)

	# Create DataLoader
	dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

	# Initialize model
	model_args = ModelArgs(
		dim=128,
		n_layers=32,
		n_heads=8,
		vocab_size=tokenizer.vocab_size + 2,
		max_seq_len=args.max_seq_len,
	)

	sample = torch.randint(0, 10, (args.batch_size, args.max_seq_len))
	model = Llama(model_args)
	if rank == 0:
		print(
			"# of trainable parameters : ",
			pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
		)
	pipe = Pipeline(model, sample, partition="metis", schedule="1f1b")

	# Initialize optimizer
	optimizer = torch.optim.AdamW(pipe.parameters(), lr=args.lr)

	# Training loop
	model.train()
	for epoch in range(args.epochs):
		total_loss = 0
		for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
			# Transform batch["input_ids"] into a single tensor
			input_ids = torch.stack(batch["input_ids"], -1).cuda()
			# attention_mask = batch["attention_mask"].cuda()

			optimizer.zero_grad()

			outputs, loss = pipe(input_ids, input_ids, model.loss, split_size=args.batch_size // 4)
			# outputs = model(input_ids)
			# loss = model.loss(outputs, input_ids)

			optimizer.step()

			if rank == ws - 1:
				total_loss += loss.item()

		if rank == ws - 1:
			avg_loss = total_loss / len(dataloader)
			print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	ws = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	main()

	if dist.is_initialized():
		dist.destroy_process_group()
