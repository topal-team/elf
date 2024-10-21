import os
import torch
import torch.distributed as dist


def init_dist():
	local_rank = int(os.getenv("LOCAL_RANK"))
	rank = int(os.getenv("RANK"))
	dist.init_process_group(backend="nccl")
	torch.cuda.set_device(local_rank)
	ws = dist.get_world_size()
	return local_rank, rank, ws


# Crashes at process group destroy with world size > 2
def run_p2p(local_rank, rank, ws):
	if rank != 0:
		x = torch.randn(10, device=local_rank)
		dist.recv(x, src=rank - 1)
	if rank != ws - 1:
		y = torch.randn(10, device=local_rank)
		dist.send(y, dst=rank + 1)

	print(f"Rank {rank} - done")

	dist.destroy_process_group()


# Always works
def run_collective(local_rank, rank, ws):
	x = torch.randn(10, device=local_rank)
	x_list = None
	if rank == 0:
		x_list = [torch.empty(10, device=local_rank) for _ in range(ws)]
	dist.gather(x, x_list, dst=0)
	print(f"Rank {rank} - {x_list}")

	dist.destroy_process_group()

# With a tensor, crashes with any world size at communication (before process group destroy)
# With something else, works fine with any world size
def run_gather_object(local_rank, rank, ws):
	# object = {rank: "hello"}
	object = {rank: torch.randn(10, device=local_rank)}
	object_list = []
	if rank == 0:
		object_list = [{} for _ in range(ws)]
	dist.gather_object(object, object_list, dst=0)
	if rank == 0:
		print(object_list)

	dist.destroy_process_group()

# With a tensor, crashes with any world size at communication (before process group destroy)
# With something else, crashes when world size > 2 at process group destroy
def run_p2p_object(local_rank, rank, ws):
	object = {rank: "hello"}
	# object = {rank: torch.randn(10, device=local_rank)}
	object_list = [{}]
	if rank != ws - 1:
		dist.send_object_list([object], dst=rank + 1)
	if rank != 0:
		dist.recv_object_list(object_list, src=rank - 1)

	print(f"Rank {rank} - {object_list}")

	dist.destroy_process_group()


if __name__ == "__main__":
	local_rank, rank, ws = init_dist()
	run_p2p(local_rank, rank, ws)
	# run_collective(local_rank, rank, ws)
	# run_gather_object(local_rank, rank, ws)
	# run_p2p_object(local_rank, rank, ws)
