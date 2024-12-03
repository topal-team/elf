import os
import torch
import torch.distributed as dist
from datetime import timedelta


def init_dist():
	local_rank = int(os.getenv("LOCAL_RANK"))
	rank = int(os.getenv("RANK"))
	dist.init_process_group(backend="nccl", timeout=timedelta(seconds=30))
	torch.cuda.set_device(local_rank)
	ws = dist.get_world_size()
	dist.barrier()  # init all communicators
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

	dist.barrier()
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

def run_multinode_test(local_rank, rank, ws):
	print(f"Global rank {rank} - local rank {local_rank}, world size {ws}")

	members = list(range(ws))
	if rank == 0:
		print(f"Members: {members}")
	group = dist.new_group(ranks=members)

	x = torch.full((2, 3), rank, device=local_rank)
	if rank != 0:
		dist.recv(x, src=rank - 1, group=group)
	if rank != ws - 1:
		dist.send(x, dst=rank + 1, group=group)

	print(f"Rank {rank} - {x}")
	torch.cuda.synchronize()
	dist.barrier(group=group)

	if rank in members:
		dist.destroy_process_group(group)

	dist.destroy_process_group()


def run_batched_test(local_rank, rank, ws):
	x = torch.randn(10, device=local_rank)
	y = torch.randn_like(x)

	print(f"Rank {rank} - starting")
	if rank == 0:
		op1 = dist.P2POp(dist.isend, x, peer=1)
		op2 = dist.P2POp(dist.irecv, y, peer=1)
		works = dist.batch_isend_irecv([op1, op2])
		for w in works:
			w.wait()
	elif rank == 1:
		ops = [dist.P2POp(dist.isend, x, peer=0), dist.P2POp(dist.irecv, y, peer=0)]
		works = dist.batch_isend_irecv(ops)
		for w in works:
			w.wait()
	else:
		return

	torch.cuda.synchronize()

	print(f"Rank {rank} - finished")
	dist.destroy_process_group()


def deadlock(local_rank, rank, ws):
	x = torch.randn(10000, 10000, device=local_rank)
	y = torch.randn_like(x)

	if rank == 0:
		wrecv = dist.irecv(y, src=1)
		wsend = dist.isend(x, dst=1)
	if rank == 1:
		wsend = dist.isend(y, dst=0)
		wrecv = dist.irecv(x, src=0)

	wrecv.wait()
	wsend.wait()
	torch.cuda.synchronize()

	print(f"Rank {rank} - Finished")
	dist.destroy_process_group()


def multiple_sends(local_rank, rank, ws):
	x = torch.randn(10, device=local_rank)
	if rank == 0:
		w1 = dist.isend(x, dst=1)
		w2 = dist.isend(x, dst=2)
		w1.wait()
		w2.wait()
	if rank == 1:
		dist.irecv(x, src=0).wait()

	torch.cuda.synchronize()
	print(f"Rank {rank} - Finished")
	dist.destroy_process_group()


if __name__ == "__main__":
	local_rank, rank, ws = init_dist()
	# run_p2p(local_rank, rank, ws)
	# run_collective(local_rank, rank, ws)
	# run_gather_object(local_rank, rank, ws)
	# run_p2p_object(local_rank, rank, ws)
	# run_multinode_test(local_rank, rank, ws)
	# run_batched_test(local_rank, rank, ws)
	# deadlock(local_rank, rank, ws)
	multiple_sends(local_rank, rank, ws)
