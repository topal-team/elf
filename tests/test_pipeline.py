import os

import torch
import torch.nn as nn
import torch.distributed as dist

from pipeline.pipeline import *

from datetime import timedelta

import pytest


@pytest.fixture(scope="session")
def init_dist():
	assert "RANK" in os.environ, "Cannot run multi-process tests without torchrun"

	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	world_size = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	try:
		if not dist.is_initialized():
			dist.init_process_group(
				backend="nccl", timeout=timedelta(seconds=60), device_id=torch.device(local_rank)
			)

		yield rank, local_rank, world_size
	finally:
		if dist.is_initialized():
			dist.destroy_process_group()


class FakeWorker:
	def __init__(self, done):
		self.done = done

	def wait(self):
		while not self.done:
			pass
		return

	def is_completed(self):
		return self.done


@pytest.mark.multi
def test_pipeline_creation_multi(init_dist):
	rank, local_rank, world_size = init_dist

	if world_size < 4:
		pytest.skip("This test needs 4 gpus (and 4 processes) to run.")
		return

	# Test automatic placement + partitioning
	sample = torch.randn((4, 10), device=local_rank)
	model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(world_size * 2)])
	# Default
	pipe = Pipeline(model, sample)
	assert len(pipe.blocks) == 1  # On every device, there is 1 block
	assert pipe.blocks[0].id == pipe.blocks[0].rank == rank

	# Test predefined placement
	pipe = Pipeline(model, sample, placement=[0, 1, 2, 3])
	assert len(pipe.blocks) == 1

	pipe = Pipeline(model, sample, placement=[1, 2, 1, 2, 0, 3], partitioner="naive")
	if rank == 1 or rank == 2:
		assert len(pipe.blocks) == 2
	else:
		assert len(pipe.blocks) == 1

	pipe = Pipeline(model, sample, placement=[1, 2, 2, 3, 0, 3], partitioner="naive")
	if rank == 0 or rank == 1:
		assert len(pipe.blocks) == 1
	else:
		assert len(pipe.blocks) == 2

	# Handmade placement
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

	# Handmade partition
	model = nn.Linear(1, 1)
	sources, targets = get_sources_targets_sequential([0, 1, 2, 3])
	pipe = Pipeline([model], sample, partitioner=False, sources=sources, targets=targets)
	assert len(pipe.blocks) == 1
	assert pipe.blocks[0].id == pipe.blocks[0].rank == rank


@pytest.mark.multi
def test_pipe_correctness_multi(init_dist):
	rank, local_rank, world_size = init_dist

	if world_size < 4:
		pytest.skip("This test needs at least 4 gpus (and processes) to run.")
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
	targets = torch.randn((4, 3), device=local_rank)
	y, loss = pipe(inputs, targets, loss_fn=nn.functional.mse_loss)
	if rank == last:
		dist.send(y, 0)
		dist.send(loss, 0)
		dist.send(targets, 0)
	elif rank == 0:
		y = torch.empty_like(targets)
		loss = torch.empty((1), device=local_rank)
		dist.recv(y, last)
		dist.recv(loss, last)
		dist.recv(targets, last)

	# Everything should be cleared after a full pass
	for b in pipe.blocks:
		for var in b.inputs:
			assert len([w for w in var.waiting if w is not None]) == 0
			assert len([k for k in var.kept if k is not None]) == 0
			assert len([f for f in var.finished if f is not None]) == 0
		for var in b.outputs:
			for target in var:
				assert len([w for w in target.waiting if w is not None]) == 0
				assert len([k for k in target.kept if k is not None]) == 0
				assert len([f for f in target.finished if f is not None]) == 0

	all_weights = [torch.empty_like(model.weight) for _ in range(world_size)] if rank == 0 else None
	all_grads = [torch.empty_like(model.weight) for _ in range(world_size)] if rank == 0 else None
	dist.gather(model.weight, all_weights, dst=0)
	dist.gather(model.weight.grad, all_grads, dst=0)
	if rank == 0:
		model_full = nn.Sequential()
		for w in all_weights:
			layer = nn.Linear(3, 3, bias=False)
			layer.weight.data = w.data
			model_full.append(layer)

		y_true = model_full(inputs)
		assert torch.allclose(y, y_true)

		loss_true = nn.functional.mse_loss(y_true, targets)
		assert torch.allclose(loss_true, loss)

		loss_true.backward()

		for grad, layer in zip(all_grads, model_full.children()):
			assert torch.allclose(grad.data, layer.weight.grad.data)


@pytest.mark.single
def test_get_mb_sizes():
	pipe = Pipeline.__new__(Pipeline)

	batch = [torch.empty(32, 1)]
	split_size = 8
	mb_sizes = pipe._get_mb_sizes(split_size, batch)
	assert mb_sizes == [8, 8, 8, 8]

	batch = [torch.empty(33, 1)]
	mb_sizes = pipe._get_mb_sizes(split_size, batch)
	assert mb_sizes == [8, 8, 8, 8, 1]

	batch = [torch.empty(32, 1)]
	split_size = [5, 9, 11, 7]
	mb_sizes = pipe._get_mb_sizes(split_size, batch)
	assert mb_sizes == [5, 9, 11, 7]
