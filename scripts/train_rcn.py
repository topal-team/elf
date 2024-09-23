import os
import sys

sys.path.append("./")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

# from models.rcn import revcol_base
from timm.models.nfnet import dm_nfnet_f4
from pipeline import Pipeline
import torch.distributed as dist
import argparse

import logging

logger = logging.getLogger("train_rcn")
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
num_epochs = 3
batch_size = 32
learning_rate = 0.001

# Define data transforms
transform = transforms.Compose(
	[
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
	]
)

# Load CIFAR-100 dataset
train_dataset = datasets.CIFAR10(root="/data", train=True, download=False, transform=transform)
train_loader = DataLoader(
	train_dataset,
	batch_size=batch_size,
	num_workers=2,
	sampler=DistributedSampler(
		train_dataset, num_replicas=args.dp, rank=rank // args.pp, shuffle=True
	),
)

# Initialize the RCN model
# model = RCN(in_channels=3, hidden_channels=64, num_blocks=8, num_columns=2)
# model = revcol_base(save_memory=False, num_classes=10)
model = dm_nfnet_f4(pretrained=False)
model.train()
if rank == 0:
	print(
		"# of trainable parameters : ",
		pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
	)

sample = torch.randn((batch_size, 3, 224, 224))
placement = list(range(args.pp)) * 2
pipe = Pipeline(model, sample, schedule="1f1b", partition="metis", dp=args.dp)

# Define loss function and optimizer
optimizer = optim.Adam(pipe.parameters(), lr=learning_rate)
# Training loop
for epoch in range(num_epochs):
	running_loss = 0.0
	for i, (inputs, labels) in enumerate(train_loader):
		inputs, labels = inputs.cuda(), labels.cuda()

		optimizer.zero_grad()
		_, loss = pipe(
			inputs, labels, loss_fn=nn.functional.cross_entropy, split_size=batch_size // args.pp
		)
		optimizer.step()

		if rank == placement[-1]:
			running_loss += loss.detach().item()
			if i % 100 == 99:  # print every 100 mini-batches
				print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (100 * batch_size):.3f}")
				running_loss = 0.0
				print(f"Last pipe time recorded : {pipe.times['total']:.3f}s")
				print(f"Max memory allocated : {torch.cuda.max_memory_allocated() / (1024 ** 2):.3f}MB")
				torch.cuda.reset_peak_memory_stats()

pipe.save("nfnet.pt", worker=0)
if rank == 0:
	print("Finished Training")
	print("Model saved to rcn.pt")

pipe.clear()
if dist.is_initialized():
	dist.destroy_process_group()
