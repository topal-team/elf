import os
import sys

sys.path.append("./")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.rcn import RCN
from tqdm import tqdm
from pipeline import Pipeline
import torch.distributed as dist


def pretty_print_params(n):
	if n > 1e9:
		return f"{n/1e9:.1f}B"
	elif n > 1e6:
		return f"{n/1e6:.1f}M"
	elif n > 1e3:
		return f"{n/1e3:.1f}K"
	else:
		return f"{int(n)}"


rank = int(os.getenv("RANK"))
local_rank = int(os.getenv("LOCAL_RANK"))
ws = int(os.getenv("WORLD_SIZE"))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

# Define hyperparameters
num_epochs = 10
batch_size = 16
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Initialize the RCN model
model = RCN(in_channels=3, hidden_channels=128, num_blocks=16, num_columns=4)
if rank == 0:
	print(
		"# of trainable parameters : ",
		pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
	)

model.train()
sample = torch.randn((4, 3, 224, 224))
pipe = Pipeline(model, sample, schedule="1f1b", partition="metis")

# Define loss function and optimizer
optimizer = optim.Adam(pipe.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
	running_loss = 0.0
	for i, (inputs, labels) in enumerate(tqdm(train_loader)):
		inputs, labels = inputs.cuda(), labels.cuda()

		optimizer.zero_grad()
		outputs, loss = pipe(
			inputs, labels, loss_fn=nn.functional.cross_entropy, split_size=batch_size // 4
		)
		optimizer.step()

		if rank == ws - 1:
			running_loss += loss.item()
			if i % 200 == 199:  # print every 200 mini-batches
				print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
				running_loss = 0.0

if rank == 0:
	print("Finished Training")

	# Save the model
	torch.save(model.state_dict(), "rcn_cifar100.pth")

if dist.is_initialized():
	dist.destroy_process_group()
