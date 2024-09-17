import sys

sys.path.append("./")
import torch
import torch.nn as nn
import torch.distributed as dist
from pipeline import Pipeline
import os


# Define a dummy PyTorch model
class DummyModel(nn.Module):
	def __init__(self):
		super(DummyModel, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(100, 256),
			nn.ReLU(),
			nn.Linear(256, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 10),
		)

	def forward(self, x):
		return self.layers(x)


if __name__ == "__main__":
	# Initialize the distributed environment
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	ws = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	# Create the model and sample input
	model = DummyModel()
	sample = torch.randn((32, 100)).cuda()

	layers_per_device = len(model.layers) // ws
	parts = [model.layers[rank * layers_per_device : (rank + 1) * layers_per_device]]

	# Create the pipeline
	pipe = Pipeline(parts, sample, partition=False)

	# Define loss function and optimizer
	loss_fn = nn.functional.cross_entropy
	optimizer = torch.optim.Adam(pipe.parameters())

	# Training loop
	num_epochs = 10
	for epoch in range(num_epochs):
		if rank == 0:
			print(f"Epoch {epoch + 1}/{num_epochs}")

		# Generate dummy data
		inputs = torch.randn((32, 100)).cuda()
		targets = torch.randint(0, 10, (32,)).cuda()

		# Forward pass
		outputs, loss = pipe(inputs, targets, loss_fn)

		# Backward pass and optimization
		optimizer.zero_grad()
		optimizer.step()

		if rank == (ws - 1) and (epoch + 1) % 2 == 0:
			print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

	pipe.clear()
	if rank == 0:
		print("Training completed.")
	if dist.is_initialized():
		dist.destroy_process_group()
