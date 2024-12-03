import os
import time
import csv

import torch
import torch.distributed as dist


def pretty_print_params(n):
	if n >= 1e9:
		return f"{n / 1e9:.2f}B"
	elif n >= 1e6:
		return f"{n / 1e6:.2f}M"
	else:
		return f"{n:.2f}"


def send_model(model, dst):
	# Send using CPU
	dist.send_object_list([model], dst)
	torch.cuda.synchronize()

	# Then send actual tensors using GPU->GPU
	params = sorted(model.named_parameters(), key=lambda x: x[0])
	for _, param in params:
		dist.send(param.data.cuda(), dst)

	buffers = sorted(model.named_buffers(), key=lambda x: x[0])
	for _, buffer in buffers:
		dist.send(buffer.data.cuda(), dst)


def recv_model(src):
	model = [None]
	dist.recv_object_list(model, src)
	torch.cuda.synchronize()
	model = model[0].to_empty(device=torch.cuda.current_device())

	params = sorted(model.named_parameters(), key=lambda x: x[0])
	for _, param in params:
		dist.recv(param.data, src)

	buffers = sorted(model.named_buffers(), key=lambda x: x[0])
	for _, buffer in buffers:
		dist.recv(buffer.data, src)

	return model


def init_communicators(rank, ws):
	x = torch.randn(1).cuda()
	if rank == 0:
		dist.send(x, 1)
		dist.recv(x, 1)
	else:
		dist.recv(x, 0)
		dist.send(x, 0)
	torch.cuda.synchronize()
	print(f"Rank {rank}: initialized communicators")


def init_model(rank, size):
	if rank == 0:
		model = torch.nn.Sequential(
			torch.nn.Linear(size, size * 2),
			torch.nn.ReLU(),
			torch.nn.Linear(size * 2, size * 4),
			torch.nn.ReLU(),
			torch.nn.Linear(size * 4, size * 2),
			torch.nn.ReLU(),
			torch.nn.Linear(size * 2, size),
		)
	else:
		model = None

	return model


def native(rank, model):
	if rank == 0:
		model.cuda()
		dist.send_object_list([model], 1)
	else:
		model = [None]
		dist.recv_object_list(model, 0)
		model = model[0].cuda()

	return model


def custom(rank, model):
	if rank == 0:
		print(
			"Rank 0 # of trainable parameters : ",
			pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
		)
		send_model(model, 1)
	else:
		model = recv_model(0)
		print(
			"Rank 1 # of trainable parameters : ",
			pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
		)

	return model


def test_models_equal(rank, model):
	model.cuda()
	x = torch.randn((16, model[0].in_features), device=torch.cuda.current_device())
	dist.broadcast(x, 0)
	y = model(x)
	print(f"Rank {rank}: model output : {y.mean().item()}")


if __name__ == "__main__":
	local_rank = int(os.environ.get("LOCAL_RANK", 0))
	torch.cuda.set_device(local_rank)
	dist.init_process_group("nccl")
	rank = dist.get_rank()
	ws = dist.get_world_size()

	init_communicators(rank, ws)

	sizes = []
	native_times = []
	custom_times = []

	size = 32
	while size <= 4096:
		if rank == 0:
			print(f"\n== Size {size} ==\n")
			sizes.append(size)
		model = init_model(rank, size)

		dist.barrier()
		start = time.time()
		model = native(rank, model)
		torch.cuda.synchronize()
		dist.barrier()
		native_times.append(time.time() - start)

		test_models_equal(rank, model)

		model = init_model(rank, size)

		dist.barrier()
		start = time.time()
		model = custom(rank, model)
		torch.cuda.synchronize()
		dist.barrier()
		custom_times.append(time.time() - start)

		torch.cuda.synchronize()
		test_models_equal(rank, model)

		size += 32

	if rank == 0:
		with open("bench-sendrecv-modules.csv", "w") as f:
			writer = csv.writer(f)
			writer.writerow(["Size", "Native Time (s)", "Custom Time (s)"])
			for size, n_time, c_time in zip(sizes, native_times, custom_times):
				writer.writerow([size, n_time, c_time])

	dist.destroy_process_group()
