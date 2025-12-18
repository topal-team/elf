import os
import sys

sys.path.append("./")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import timm

from elf import Pipeline
from elf.utils import pretty_print_params
from configmypy import YamlConfig, ArgparseConfig

import logging

logger = logging.getLogger("train_vision")
logging.basicConfig(level=logging.INFO)


def get_dataset(config):
	# Define data transforms
	transform = transforms.Compose(
		[
			transforms.Resize(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		]
	)

	match config.data.dataset:
		case "cifar100":
			dataset = datasets.CIFAR100(
				root=config.data.dataset_dir, train=True, download=True, transform=transform
			)
		case "cifar10":
			dataset = datasets.CIFAR10(
				root=config.data.dataset_dir, train=True, download=True, transform=transform
			)
		case _:
			raise ValueError(
				f"Unknown dataset: {config.data.dataset}. Available datasets: cifar100, cifar10"
			)

	return dataset


def parse_config():
	config, _ = YamlConfig("./config/train_vision.yaml", config_name="default").read_conf()
	config, _ = ArgparseConfig().read_conf(config)
	return config


if __name__ == "__main__":
	config = parse_config()

	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	ws = int(os.getenv("WORLD_SIZE"))

	assert config.pipeline.dp * config.pipeline.pp == ws, "dp * pp must be equal to world size"

	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

	# Define hyperparameters
	num_epochs = config.train.epochs
	batch_size = config.train.batch_size
	learning_rate = config.train.learning_rate

	train_dataset = get_dataset(config)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		sampler=DistributedSampler(
			train_dataset, num_replicas=config.pipeline.dp, rank=rank // config.pipeline.pp, shuffle=True
		),
	)

	model = timm.create_model(
		config.model.arch, pretrained=config.model.pretrained, num_classes=config.model.num_classes
	)
	model.train()
	if rank == 0:
		print(
			"# of trainable parameters : ",
			pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
		)

	sample = torch.randn((batch_size // config.pipeline.pp, 3, 224, 224))
	pipe = Pipeline(
		model,
		sample,
		placement=config.pipeline.placement,
		scheduler=config.pipeline.scheduler_type,
		dp=config.pipeline.dp,
	)

	# Define loss function and optimizer
	optimizer = optim.Adam(pipe.parameters(), lr=learning_rate)
	# Training loop
	running_loss = 0.0
	step = 0
	for epoch in range(num_epochs):
		for i, (inputs, labels) in enumerate(train_loader):
			step += 1
			inputs, labels = inputs.cuda(), labels.cuda()

			optimizer.zero_grad()
			_, loss = pipe(inputs, labels, loss_fn=nn.functional.cross_entropy)
			optimizer.step()
			if loss:
				running_loss += loss.item()

			if step == 10:
				torch.cuda.reset_peak_memory_stats()
				if rank == pipe.placement[-1]:
					running_loss /= step
					print(f"\nLoss: {running_loss:.3f}\n")

					running_loss = 0.0
				step = 0

	pipe.save("vision.pt", worker=0)
	if rank == 0:
		print("Finished Training")
		print("Model saved to vision.pt")

	pipe.clear()
	if dist.is_initialized():
		dist.destroy_process_group()
