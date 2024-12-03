import argparse
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
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

from models.GPT import (
	GPT,
	MyGPTConfig,
	GPTSmallConfig,
	GPT13BConfig,
	GPT175BConfig,
	GPTXXLConfig,
)
from pipeline import Pipeline

import logging
import json

logger = logging.getLogger("train_llama")
logging.basicConfig(level=logging.INFO)


class DummyGPT(nn.Module):
	def __init__(self, n_layers=16):
		super().__init__()
		self.n_layers = n_layers
		setattr(self, f"layer{1}", nn.Linear(3072, 512))
		for i in range(n_layers - 1):
			setattr(self, f"layer{i}", nn.Linear(512, 512))

	def forward(self, x):
		for i in range(self.n_layers):
			x = self.__getattr__(f"layer{i}")(x)
		return x


def parse_args():
	parser = argparse.ArgumentParser()
	# Phase
	parser.add_argument(
		"--phase", type=str, help="Pipeline phase", choices=["prepare_data", "prepare_model", "train"]
	)
	parser.add_argument("--save_dir", type=str)

	# Data
	parser.add_argument(
		"--data.load_dataset", action=argparse.BooleanOptionalAction, help="Dataset loading"
	)
	parser.add_argument(
		"--data.dataset_dir", type=str, help="Data: path to the folder with the dataset"
	)
	parser.add_argument(
		"--data.dataset_name", type=str, help="Data: dataset name", choices=["c4realnewslike"]
	)
	parser.add_argument("--data.c4realnewslike.dataset_dir", type=str)
	parser.add_argument(
		"--data.dataset_size", type=str, help="Data: path to the folder with the dataset"
	)

	# Model
	parser.add_argument("--model", type=str, choices=["resnet", "gpt"], default="gpt")
	parser.add_argument("--resnet.arch", type=str, choices=["resnet18", "resnet50"])
	parser.add_argument(
		"--gpt.arch", type=str, choices=["GPTSmall", "GPTLarge", "GPTXXL", "GPT13B", "GPT175B"]
	)

	# Training hyperparameters
	parser.add_argument(
		"--train.learning_rate", type=float, default=1e-2, help="Training: learning rate"
	)
	parser.add_argument(
		"--train.weight_decay", type=float, default=1e-5, help="Training: weight decay"
	)
	parser.add_argument("--train.batch_size", type=int, default=32, help="Training: batch size")
	parser.add_argument("--train.epochs", type=int, default=5, help="Training: number of epochs")
	parser.add_argument(
		"--train.max_seq_len", type=int, default=512, required=False, help="Maximum sequence length"
	)

	# Testing hyperparameters
	parser.add_argument("--test.batch_size", type=int, default=32, help="Testing: batch size")
	parser.add_argument("--test.epochs", type=int, default=5, help="Testing: number of epochs")

	# Pipeline schedule hyperparameters
	parser.add_argument(
		"--pipeline.pp",
		type=int,
		default=4,
		help="Pipeline: PP degree, number of stages in pipeline parallelism",
	)
	parser.add_argument(
		"--pipeline.dp",
		type=int,
		default=1,
		help="Pipeline: DP degree, number of model copies to process in parallel",
	)
	parser.add_argument(
		"--pipeline.schedule_type",
		type=str,
		default="afab",
		help="Pipeline: default schedule to use",
		choices=["1f1b", "afab"],
	)
	parser.add_argument(
		"--pipeline.pp_placement",
		type=str,
		default="0,1,2,3",
		help='Pipeline: pp_placement="0,1,2,3" for afab/1f1b, =[0,1,2,3,0,1,2,3] for interleaved 1f1b',
	)
	parser.add_argument(
		"--pipeline.partition",
		type=str,
		default="metis",
		help="Pipeline: partition",
		choices=["metis", "naive"],
	)

	parser.add_argument(
		"--log", choices=["debug", "info", "none"], default="info", required=False, help="logging level"
	)
	parser.add_argument("--slurm_jobid", type=int, default=0)

	return parser.parse_args()


def merge_args():
	dataset_config = YamlConfig(
		"./dataset_config.yaml", config_name="default", config_folder="./config"
	)
	train_test_config = YamlConfig(
		"./train_test_config.yaml", config_name="default", config_folder="./config"
	)
	pipeline_config = YamlConfig(
		"./pipeline_config.yaml", config_name="default", config_folder="./config"
	)
	args = parse_args()
	model_config = YamlConfig(
		f"./{args.model}_config.yaml", config_name="default", config_folder="./config"
	)

	config_pipe = ConfigPipeline(
		[dataset_config, model_config, train_test_config, pipeline_config, ArgparseConfig()]
	)
	args = config_pipe.read_conf()
	return args


def prepare_data(args):
	print(args.data)
	if not os.path.exists(args.data.__getattr__(f"{args.data.dataset_name}").dataset_dir):
		raise Exception("Dataset is not loaded")
	else:
		print("Dataset is loaded")


def prepare_tokenizer(args):
	if not os.path.exists(f"{args.tokenizer_dir}"):
		raise Exception("Tokenizer is not loaded")
	else:
		print("Tokenizer is loaded")
		if not os.path.exists(
			f'{args.data.__getattr__(f"{args.data.dataset_name}").dataset_dir}/tokenized/train'
		):
			raise Exception("Tokenized dataset is not found")
		else:
			print("Tokenized dataset is loaded")


def pretty_print_params(n):
	if n > 1e9:
		return f"{n/1e9:.1f}B"
	elif n > 1e6:
		return f"{n/1e6:.1f}M"
	else:
		return f"{int(n)}"


def get_gpt_config(args, vocab_size):
	match args.gpt.arch:
		case "GPTSmall":
			return MyGPTConfig(vocab_size + 2, args.train.max_seq_len, n_layer=12, n_head=12, n_embd=768)
			# return GPTSmallConfig(vocab_size + 2, args.train.max_seq_len)
		case "GPTLarge":
			return GPTSmallConfig(vocab_size + 2, args.train.max_seq_len)
		case "GPTXXL":
			return GPTXXLConfig(vocab_size + 2, args.train.max_seq_len)
		case "GPT13B":
			return GPT13BConfig(vocab_size + 2, args.train.max_seq_len)
		case "GPT175B":
			return GPT175BConfig(vocab_size + 2, args.train.max_seq_len)


def main(args):
	rank = dist.get_rank()

	tokenizer = AutoTokenizer.from_pretrained(f"{args.tokenizer_dir}")
	tokenizer.pad_token = tokenizer.eos_token
	tokenized_dataset = datasets.load_from_disk(
		args.data.__getattr__(f"{args.data.dataset_name}").dataset_dir + "/tokenized/train"
	)
	tokenized_dataset = tokenized_dataset.select(range(args.data.dataset_size))
	if rank == 0:
		print(f"Loaded {pretty_print_params(len(tokenized_dataset))} samples")
		print(f"Vocab size: {tokenizer.vocab_size}")

	# Create DataLoader
	dataloader = DataLoader(
		tokenized_dataset,
		batch_size=args.train.batch_size // args.pipeline.dp,
		sampler=DistributedSampler(
			tokenized_dataset, num_replicas=args.pipeline.dp, rank=rank // args.pipeline.pp
		),
	)

	sample = torch.randint(0, 10, (args.train.batch_size // args.pipeline.pp, args.train.max_seq_len))

	model = GPT(get_gpt_config(args, tokenizer.vocab_size))

	if rank == 0:
		print(
			"# of trainable parameters : ",
			pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
		)

	placement = list(map(int, args.pipeline.pp_placement.strip().split(",")))
	# placement = list(range(args.pipeline.pp)) * 2
	# args.pipeline.partition = False
	pipe = Pipeline(
		model,
		sample,
		placement,
		partition=args.pipeline.partition,
		schedule=args.pipeline.schedule_type,
		dp=args.pipeline.dp,
	)

	# Initialize optimizer
	pipe.optimizer = torch.optim.AdamW(
		pipe.parameters(), lr=args.train.learning_rate, weight_decay=args.train.weight_decay
	)

	def loss_fn(logits, targets):
		logits = logits.view(-1, logits.size(-1))
		targets = targets.view(-1)
		return nn.functional.cross_entropy(logits, targets)

	torch.cuda.cudart().cudaProfilerStart()

	# save model before training
	pipe.save_model("init", checkpoints_dir=f"{args.save_dir}/{args.slurm_jobid}")

	# Training loop
	model.train()
	for epoch in range(args.train.epochs):
		total_loss = 0
		i = 0
		for i, batch in enumerate(dataloader):
			if i > 200:
				break
			# Transform batch["input_ids"] into a single tensor
			input_ids = torch.stack(batch["input_ids"], -1).cuda()

			pipe.optimizer.zero_grad()

			_, loss = pipe(
				input_ids,
				input_ids,
				loss_fn,
				split_size=args.train.batch_size // args.pipeline.pp,
				profile=True,
			)

			pipe.optimizer.step()

			if rank == world_size - 1:
				total_loss += loss.item()
				if i % 100 == 99:
					print(f"[{epoch + 1} | {i}] : {total_loss / i:.4f}")

		if rank == world_size - 1:
			avg_loss = total_loss / len(dataloader)
			print(f"Epoch {epoch + 1}/{args.train.epochs}, Average Loss: {avg_loss:.7f}")

		pipe.save_state_dict(epoch + 1, f"{args.save_dir}/{args.slurm_jobid}")

	torch.cuda.cudart().cudaProfilerStop()
	pipe.clear()


def init_process(local_rank, fn, backend="nccl"):
	"""Initialize the distributed environment."""
	dist.init_process_group(backend, rank=local_rank, init_method="env://")
	size = dist.get_world_size()
	if torch.cuda.is_available() and backend == "nccl":
		torch.cuda.set_device(local_rank)
	fn(local_rank, size)


if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	world_size = int(os.getenv("WORLD_SIZE"))

	dist.init_process_group(backend="nccl", rank=rank)
	torch.cuda.set_device(local_rank)

	args = merge_args()

	if rank == 0:
		print(f"ConfigPipeline.read_conf() output:\n {args}")
		savedir = f"{args.save_dir}/{args.slurm_jobid}"
		os.makedirs(savedir, exist_ok=True)

		with open(f"{savedir}/args_launch.json", "w") as f:
			json.dump(args, f)

		if args.phase == "prepare_data":
			prepare_data(args)
		elif args.phase == "prepare_model":
			prepare_tokenizer(args)
		else:
			pass

	main(args)

	if dist.is_initialized():
		dist.destroy_process_group()
