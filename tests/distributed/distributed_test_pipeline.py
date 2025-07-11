import sys

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.append(".")
from elf import Pipeline, get_sources_targets_sequential

from tests.distributed.distributed_utils import test


class FakeWorker:
	def __init__(self, done):
		self.done = done

	def wait(self):
		while not self.done:
			pass
		return

	def is_completed(self):
		return self.done


def test_pipeline_creation(rank, local_rank, world_size):
	if world_size < 4:
		print("This test needs 4 gpus (and 4 processes) to run.")
		return

	sample = torch.randn((4, 64), device=local_rank)
	model = nn.Sequential(*[nn.Linear(64, 64) for _ in range(world_size * 2)])

	pipe = Pipeline(model, sample)
	assert len(pipe.blocks) == 1
	assert pipe.blocks[0].id == pipe.blocks[0].rank == rank

	pipe = Pipeline(model, sample, placement=[0, 1, 2, 3])
	assert len(pipe.blocks) == 1

	pipe = Pipeline(model, sample, placement=[1, 2, 1, 2, 0, 3], partitioner="naive")
	if rank in (1, 2):
		assert len(pipe.blocks) == 2
	else:
		assert len(pipe.blocks) == 1

	pipe = Pipeline(model, sample, placement=[1, 2, 2, 3, 0, 3], partitioner="naive")
	if rank in (0, 1):
		assert len(pipe.blocks) == 1
	else:
		assert len(pipe.blocks) == 2

	pipe = Pipeline(model, sample, placement=[0, 1, 2, 3])
	if rank == 0:
		assert len(pipe.blocks) == 1
		b1 = pipe.blocks[0]
		assert b1.rank == rank
		assert b1.id == 0
		assert b1.is_first
	elif rank == 1:
		assert len(pipe.blocks) == 1
		b1 = pipe.blocks[0]
		assert b1.rank == rank
		assert b1.id == 1

	model = nn.Linear(1, 1)
	sources, targets = get_sources_targets_sequential([0, 1, 2, 3])
	pipe = Pipeline([model], sample, partitioner=False, sources=sources, targets=targets)
	assert len(pipe.blocks) == 1
	assert pipe.blocks[0].id == pipe.blocks[0].rank == rank


def test_pipe_correctness(rank, local_rank, world_size):
	if world_size < 4:
		print("This test needs at least 4 gpus (and processes) to run.")
		return

	sample = torch.randn((4, 3), device=local_rank)
	model = nn.Linear(3, 3, bias=False).cuda()
	placement = [0, 1, 2, 3]
	sources, targets = get_sources_targets_sequential(placement)

	pipe = Pipeline(
		[model], sample, placement=placement, partitioner=False, sources=sources, targets=targets
	)
	last = 3

	inputs = torch.randn((4, 3), device=local_rank)
	targets_t = torch.randn((4, 3), device=local_rank)
	y, loss = pipe(inputs, targets_t, loss_fn=nn.functional.mse_loss)
	if rank == last:
		dist.send(y, 0)
		dist.send(loss, 0)
		dist.send(targets_t, 0)
	elif rank == 0:
		y_recv = torch.empty_like(targets_t)
		loss_recv = torch.empty((1), device=local_rank)
		dist.recv(y_recv, last)
		dist.recv(loss_recv, last)
		dist.recv(targets_t, last)

	for b in pipe.blocks:
		for var in b.input_variables:
			assert len([w for w in var.to_process if w is not None]) == 0
			assert len([k for k in var.saved if k is not None]) == 0
			assert len([f for f in var.to_send if f is not None]) == 0
		for var in b.output_variables:
			for target in var:
				assert len([w for w in target.to_process if w is not None]) == 0
				assert len([k for k in target.saved if k is not None]) == 0
				assert len([f for f in target.to_send if f is not None]) == 0


def test_inference_pipeline(rank, local_rank, world_size):
	if world_size < 4:
		print("This test needs at least 4 gpus (and processes) to run.")
		return

	model = nn.Sequential(
		nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
	).cuda()
	inputs = torch.randn(4, 10).cuda()

	if rank == 0:
		ref = model(inputs)

	part = model[local_rank]
	if rank == 0:
		for i in range(1, world_size):
			dist.send(model[i].weight, i)
			dist.send(model[i].bias, i)
	else:
		dist.recv(part.weight.data, 0)
		dist.recv(part.bias.data, 0)

	sources, targets = get_sources_targets_sequential([0, 1, 2, 3])

	pipe = Pipeline(
		part, None, scheduler="inference", partitioner=False, sources=sources, targets=targets
	)
	assert len(pipe.blocks) == 1
	assert pipe.blocks[0].id == pipe.blocks[0].rank == rank

	y, _ = pipe(inputs, None, None)
	if rank == 0:
		y_recv = torch.empty_like(ref)
		dist.recv(y_recv, 3)
		assert torch.allclose(y_recv, ref)
	elif rank == 3:
		dist.send(y, 0)

	dist.barrier()


def main():
	test(test_pipeline_creation)
	test(test_pipe_correctness)
	test(test_inference_pipeline)


if __name__ == "__main__":
	main()
