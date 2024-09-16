import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import os

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
print(f"GPU {global_rank} set on local device {rank} with world size {world_size}")


class LinearBlock(nn.Module):
	def __init__(self, in_features, out_features):
		super(LinearBlock, self).__init__()
		self.layer = nn.Linear(in_features, out_features, bias=False)
		self.inputs = []
		self.activations = []

	def forward(self, x):
		# save for backward
		x.requires_grad = True
		self.inputs.append(x)
		y = self.layer(x)
		self.activations.append(y)
		return y


def pipelined_forward(layer, sample):
	splits = iter(sample.split(1, dim=0))
	result = []
	d = global_rank

	for i, x in enumerate(splits):
		# x is already on device 0
		if d != 0:
			dist.recv(x, d - 1)
			print(f"[GPU {global_rank}] - Received data from {d - 1}, starting compute")
		y = layer(x)
		if d == world_size - 1:  # last layer stores result
			result.append(y)
			print(f"[GPU {global_rank}] - Finished computing for micro batch {i}")
		else:
			# send to next layer
			print(f"[GPU {global_rank}] - Sending my data to {d + 1}")
			dist.send(y, d + 1)

	print(f"[GPU {global_rank}] - Waiting for all processes..")
	dist.barrier()
	print(f"[GPU {global_rank}] - Done !")

	if result:
		return torch.cat(result, dim=0)
	else:
		return []


def async_pipelined_forward(layer, sample):
	splits = iter(sample.split(1, dim=0))
	result = []
	d = global_rank

	for i, x in enumerate(splits):
		if d != 0:
			status = dist.irecv(x, d - 1)
			status.wait()
			print(f"[GPU {global_rank}] - Received data from rank {d - 1}")
		y = layer(x)
		if d == world_size - 1:
			print(f"[GPU {global_rank}] - Finished compute for micro batch {i}")
			result.append(y)
		else:
			print(f"[GPU {global_rank}] - Sending data to rank {d + 1}")
			dist.isend(y, d + 1)

	print(f"[GPU {global_rank}] - Waiting for all processes..")
	dist.barrier()
	if result:  # only last device has result
		return torch.cat(result, dim=0)
	else:
		return []


def interleaved_pipelined_forward(layers, sample):
	assert (
		sample.size(0) % world_size == 0
	), "# Of micro-batches should be a multiple of the number of gpus"
	n_waves = sample.size(0) // world_size
	assert n_waves == len(layers), "Each device should have as many layers as the number of waves"
	splits = iter(sample.split(1, dim=0))
	ret = []

	for i, x in enumerate(splits):
		for wave in range(n_waves):
			if global_rank != 0 and wave != 0:
				dist.recv(x, (i - 1) % world_size)
			y = layers[wave](x)
			if global_rank == world_size - 1 and wave == n_waves - 1:
				ret.append(y)
			else:
				dist.send(y, (i + 1) % world_size)

	dist.barrier()
	if result:
		return torch.cat(ret, dim=0)
	else:
		return []


def pipelined_backward(layer):
	result = []
	for i in range(len(layer.activations)):
		# Compute gradients for current block
		# Last layer has already been computed by call to loss.backward()
		if global_rank != world_size - 1:
			grads = torch.empty_like(layer.activations[i])
			dist.recv(grads, global_rank + 1)
			print(f"[GPU {global_rank}] - Received gradients from rank {global_rank + 1}")
			(
				layer.activations[i] * grads
			).sum().backward()  # hack to pass gradients : d(wx)/dx = w, here gradients of activations will be equal to "grads"
		grads = layer.inputs[i].grad.data
		if global_rank == 0:
			print(f"[GPU {global_rank}] - Finished backward pass of micro batch {i}!")
			result.append(grads)  # finished
		else:
			# Send gradients to previous block
			print(f"[GPU {global_rank}] - Sending gradients to rank {global_rank - 1}")
			dist.send(grads, global_rank - 1)

	dist.barrier()  # We can probably change the code so that this sync is not needed anymore
	if result:
		return torch.cat(result, dim=0)
	else:
		return []


if __name__ == "__main__":
	torch.cuda.set_device(
		rank
	)  # set before initiating pg otherwise cuda detects multiple processes using same device
	dist.init_process_group(backend="nccl", world_size=world_size)

	torch.manual_seed(47)

	layer = LinearBlock(3, 3).to(rank)

	# Send full model to last gpu for comparison later
	states = [None] * world_size if global_rank == world_size - 1 else None
	dist.gather_object(layer.layer.state_dict(), states, dst=world_size - 1)

	sample = torch.randn((4, 3)).to(rank)  # first layer is on rank 0

	result = async_pipelined_forward(layer, sample)

	dist.broadcast(sample, 0)  # synchronize samples for comparison later

	# Only the last rank has the result ; the others can stop
	if global_rank == world_size - 1:
		assert (
			result.shape == sample.shape
		), f"Expected result to have shape {sample.shape}, got {result.shape}"
		# Normal computation for comparison
		# Reconstruct full model on last GPU
		model = nn.Sequential(*[nn.Linear(3, 3, bias=False) for _ in range(world_size)])
		for i, state in enumerate(states):
			model[i].load_state_dict(state)

		model = model.to(rank)

		sample_full = sample.clone().detach()
		sample_full.requires_grad = True
		result_full = model(sample_full)

		assert torch.allclose(
			result, result_full
		), f"Results are different for full and pipelined models : {result_full} vs {result}"
		print("Forward pass output is correct.")

		target = torch.randn_like(result)
		loss = F.mse_loss(result, target)
		loss.backward()

		loss_full = F.mse_loss(result_full, target)
		loss_full.backward()

	grads = pipelined_backward(layer)

	if global_rank == world_size - 1:
		print("Backward pass finished for all processes.")
		print(f"[GPU {global_rank}] - Sending real grads to device 0")
		dist.send(sample_full.grad.data, 0)  # For comparison on device 0

	if global_rank == 0:
		grads_full = torch.zeros_like(sample)
		print(f"[GPU {global_rank}] - Waiting for real grads from gpu {world_size - 1}..")
		dist.recv(grads_full, world_size - 1)

		assert (
			grads_full.shape == grads.shape
		), f"Expected grads to have shape {grads_full.shape}, got {grads.shape}"

		assert torch.allclose(
			grads, grads_full
		), f"Gradients are different for full and pipelined model : {grads_full} vs {grads}"
		print("Gradients are correct.")

	print(f"[GPU {global_rank}] - Waiting for all processes..")
	dist.barrier()
	print("Closing !")
	if dist.is_initialized():
		dist.destroy_process_group()
