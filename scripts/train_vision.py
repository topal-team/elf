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

from pipeline import Pipeline

import argparse
import logging

logger = logging.getLogger("train_vision")
logging.basicConfig(level=logging.INFO)


def pretty_print_params(n):
	if n > 1e9:
		return f"{n/1e9:.1f}B"
	elif n > 1e6:
		return f"{n/1e6:.1f}M"
	elif n > 1e3:
		return f"{n/1e3:.1f}K"
	else:
		return f"{int(n)}"


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dp", type=int, default=1, help="Number of data parallel processes")
parser.add_argument("-pp", type=int, default=4, help="Number of processes in a pipeline")
parser.add_argument("-bs", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
parser.add_argument("--dataset", "-d", type=str, default="/data", help="Path to the dataset")
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


rank = int(os.getenv("RANK"))
local_rank = int(os.getenv("LOCAL_RANK"))
ws = int(os.getenv("WORLD_SIZE"))

assert args.dp * args.pp == ws, "dp * pp must be equal to world size"

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

# Define hyperparameters
num_epochs = args.epochs
batch_size = args.bs
learning_rate = 0.0005

# Define data transforms
transform = transforms.Compose(
	[
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
	]
)

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR100(root=args.dataset, train=True, download=True, transform=transform)
train_loader = DataLoader(
	train_dataset,
	batch_size=batch_size,
	sampler=DistributedSampler(
		train_dataset, num_replicas=args.dp, rank=rank // args.pp, shuffle=True
	),
)

model = timm.create_model("convnextv2_huge", pretrained=False, num_classes=100)
model.train()
if rank == 0:
	print(
		"# of trainable parameters : ",
		pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
	)

placement = list(range(args.pp)) * 2
sample = torch.randn((batch_size // len(placement), 3, 224, 224))
pipe = Pipeline(model, sample, placement, schedule="1f1b", partition="metis", dp=args.dp)

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
		_, loss = pipe(
			inputs, labels, loss_fn=nn.functional.cross_entropy, split_size=batch_size // len(placement)
		)
		optimizer.step()
		if loss:
			running_loss += loss.item()

		if step == 10:
			print(
				f"Rank {rank} - Max memory allocated : {torch.cuda.max_memory_allocated() / (2 ** 30):.3f}GB"
			)
			torch.cuda.reset_peak_memory_stats()
			if rank == placement[-1]:
				running_loss /= step
				print(f"[Epoch {epoch}, Batch {i} / {len(train_loader)}] loss: {running_loss:.3f}")
				print(f"Last pipe time recorded : {pipe.times['total']:.3f}s")

				running_loss = 0.0
			step = 0

pipe.save("vision.pt", worker=0)
if rank == 0:
	print("Finished Training")
	print("Model saved to vision.pt")

# pipe.clear()
# if dist.is_initialized():
# 	dist.destroy_process_group()
