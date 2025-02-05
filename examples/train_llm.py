import os
import sys

sys.path.append("./")

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets
from transformers import AutoTokenizer

from models.GPT import (
	GPT,
	GPTLargeConfig,
	GPTSmallConfig,
	GPTXXLConfig,
	GPT13BConfig,
	GPT175BConfig,
)

from elf import Pipeline

from configmypy import YamlConfig, ArgparseConfig
import logging

logger = logging.getLogger("train_llm")
logging.basicConfig(level=logging.INFO)


def pretty_print_params(n):
	if n > 1e9:
		return f"{n / 1e9:.1f}B"
	elif n > 1e6:
		return f"{n / 1e6:.1f}M"
	else:
		return f"{int(n)}"


def get_gpt_config(args, vocab_size):
	match args.model.arch:
		case "GPTSmall":
			return GPTSmallConfig(vocab_size + 2, args.train.max_seq_len)
		case "GPTLarge":
			return GPTLargeConfig(vocab_size + 2, args.train.max_seq_len)
		case "GPTXXL":
			return GPTXXLConfig(vocab_size + 2, args.train.max_seq_len)
		case "GPT13B":
			return GPT13BConfig(vocab_size + 2, args.train.max_seq_len)
		case "GPT175B":
			return GPT175BConfig(vocab_size + 2, args.train.max_seq_len)
		case _:
			raise ValueError(
				f"Unknown model architecture: {args.model.arch}. Available ones: GPTSmall, GPTLarge, GPTXXL, GPT13B, GPT175B"
			)


def parse_config():
	config, _ = YamlConfig("./config/train_llm.yaml", config_name="default").read_conf()
	config, _ = ArgparseConfig().read_conf(config)

	match config.log:
		case "debug":
			logging.getLogger().setLevel(logging.DEBUG)
		case "info":
			logging.getLogger().setLevel(logging.INFO)
		case "none":
			logging.getLogger().setLevel(100)

	return config


def main():
	config = parse_config()

	tokenizer = AutoTokenizer.from_pretrained(f"{config.data.tokenizer_dir}")
	tokenizer.pad_token = tokenizer.eos_token

	if os.path.exists(config.data.dataset_dir + "/tokenized/train"):
		tokenized_dataset = datasets.load_from_disk(config.data.dataset_dir + "/tokenized/train")
		tokenized_dataset = tokenized_dataset.select(range(config.data.size))
	else:
		print(f"No tokenized dataset found at {config.dataset_path}/tokenized/train.")
		exit(1)

	if rank == 0:
		print(f"Loaded {pretty_print_params(len(tokenized_dataset))} samples")
		print(f"Vocab size: {tokenizer.vocab_size}")

	# Create DataLoader
	dataloader = DataLoader(
		tokenized_dataset,
		batch_size=config.train.batch_size // config.pipeline.dp,
		sampler=DistributedSampler(
			tokenized_dataset, num_replicas=config.pipeline.dp, rank=rank // config.pipeline.pp
		),
	)

	sample = torch.randint(
		0, 10, (config.train.batch_size // config.pipeline.pp, config.train.max_seq_len)
	)
	model = GPT(get_gpt_config(config, tokenizer.vocab_size))

	if rank == 0:
		print(
			"# of trainable parameters : ",
			pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
		)

	pipe = Pipeline(
		model,
		sample,
		placement=config.pipeline.placement,
		partitioner=config.pipeline.partitioner,
		schedule=config.pipeline.schedule_type,
		dp=config.pipeline.dp,
	)

	optimizer = torch.optim.AdamW(
		pipe.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
	)

	os.makedirs(f"{config.model.checkpoint_dir}", exist_ok=True)
	pipe.checkpoint("init", dir_path=f"{config.model.checkpoint_dir}")

	if config.profile:
		torch.cuda.cudart().cudaProfilerStart()

	# Training loop
	model.train()
	for epoch in range(config.train.epochs):
		total_loss = 0
		i = 0
		for i, batch in enumerate(dataloader):
			# Transform batch["input_ids"] into a single tensor
			input_ids = torch.stack(batch["input_ids"], -1).cuda()

			optimizer.zero_grad()

			_, loss = pipe(input_ids, input_ids, model.loss_fn, profile=config.profile)

			optimizer.step()

			if rank == ws - 1:
				total_loss += loss.item()
				if i % 100 == 99:
					print(f"[{epoch + 1} | {i}] : {total_loss / i:.4f}")

		if rank == ws - 1:
			avg_loss = total_loss / len(dataloader)
			print(f"Epoch {epoch + 1}/{config.train.epochs}, Average Loss: {avg_loss:.4f}")

			pipe.checkpoint(f"epoch_{epoch + 1}", dir_path=f"{config.model.checkpoint_dir}")

	if config.profile:
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
