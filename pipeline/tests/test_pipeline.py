import torch
import torch.nn as nn
import torch.distributed as dist

from ..pipeline import *
from ..utils import TensorMetadata, dtypes

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
		dist.destroy_process_group()


@pytest.mark.single
def test_metadata():
	# Test creation from class method from_tensor
	received = torch.tensor([dtypes.index(torch.float32), 2, 3, 4])
	metadata = TensorMetadata.from_tensor(received)

	assert metadata.dtype == torch.float32
	assert metadata.shape == [2, 3, 4]

	# Test initialization by constructor
	t = torch.empty((6, 3, 2), dtype=torch.bfloat16)
	metadata = TensorMetadata(t)
	assert metadata.dtype == torch.bfloat16
	assert metadata.shape == [6, 3, 2]

	# Test tensor representation of metadata
	t = torch.empty((3, 1, 4), dtype=torch.float16)
	metadata = TensorMetadata(t)
	tensor_repr = metadata.to_tensor()
	assert tensor_repr[0] == dtypes.index(torch.float16)
	assert tensor_repr[1] == 3
	assert tensor_repr[2] == 1
	assert tensor_repr[3] == 4
	assert tensor_repr[4] == 0

	# Test buffer creation
	t = torch.empty((8, 5, 2), dtype=torch.int64)
	metadata = TensorMetadata(t)
	buffer = metadata.get_buffer(3)
	assert buffer.dtype == torch.int64
	assert buffer.shape == torch.Size([3, 8, 5, 2])


class FakeWorker:
	def __init__(self, done):
		self.done = done

	def wait(self):
		while not self.done:
			pass
		return

	def is_completed(self):
		return self.done


def _fake_p2p(tensor):
	return [FakeWorker(True), tensor]


@pytest.mark.single
def test_block():
	device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

	model = nn.Linear(2, 1, bias=False)

	# Test pipe links
	block = PipelineBlock(model, id_=0, placement=[0, 2])

	assert block.id == 0
	assert block.previous is None
	assert block.next == 2
	assert block.rank == 0

	block = PipelineBlock(model, id_=1, placement=[1, 2])
	assert block.id == 1
	assert block.previous == 1
	assert block.next is None
	assert block.rank == 2

	block = PipelineBlock(model, id_=1, placement=[2, 1, 0])
	assert block.id == 1
	assert block.previous == 2
	assert block.next == 0
	assert block.rank == 1

	# Test forward pass
	block.model.weight = nn.Parameter(torch.tensor([3.0, -1.0], device=device))
	block.metadata = [TensorMetadata(torch.tensor([2.0, 4.0], device=device))]
	block.out_metadata = [TensorMetadata(torch.tensor([1.0], device=device))]
	assert len(block.inputs_to_forward) == 0
	inputs = [_fake_p2p(torch.tensor([2.0, 4.0], device=device))]
	block.inputs_to_forward.append(inputs)

	block.forward()

	expected_result = torch.tensor([2.0], device=device)
	assert torch.allclose(block.act_to_keep[0][0], expected_result)
	assert torch.allclose(block.act_to_send[0][0], expected_result)
	assert len(block.inputs_to_forward) == 0
	assert len(block.inputs_to_keep) == 1

	# Test backward pass
	grads = [_fake_p2p(torch.tensor(3.0, device=device))]
	block.grads_to_backward.append(grads)

	block.backward()
	expected_grads_weights = torch.tensor([6.0, 12.0], device=device)
	expected_grads_inputs = torch.tensor([9.0, -3.0], device=device)

	assert torch.allclose(block.model.weight.grad.data, expected_grads_weights)
	assert torch.allclose(inputs[0].grad.data, expected_grads_inputs)

	assert len(block.act_to_keep) == 0
	assert len(block.inputs_to_keep) == 0
	assert len(block.grads_to_send) == 1


@pytest.mark.multi
def test_block_multi(init_dist):
	rank, local_rank, world_size = init_dist

	if rank > 1:
		pytest.skip("This test only needs 2 processes")

	placement = [0, 1]
	model = nn.Linear(rank + 1, rank + 2, bias=False)
	block = PipelineBlock(model, rank, placement)
	block.pp_group = None  # use default group
	block.metadata = [TensorMetadata(torch.ones((rank + 1,), device=local_rank))]
	block.out_metadata = [TensorMetadata(torch.ones((rank + 2,), device=local_rank))]

	assert block.rank == rank
	if rank == 0:
		assert block.previous is None
		assert block.next == 1

		block.model.weight = nn.Parameter(torch.tensor([[2.0], [2.0]], device=local_rank))

		assert len(block.inputs_to_forward) == 0
		w = block.recv_forward(2)  # Should do nothing as block has no previous
		assert w is None
		assert len(block.inputs_to_forward) == 0
		inputs = [_fake_p2p(torch.ones((2, 1), device=local_rank))]
		block.inputs_to_forward.append(inputs)

		assert len(block.act_to_send) == 0
		block.forward()
		assert len(block.act_to_keep) == 1
		assert len(block.act_to_send) == 1
		assert len(block.inputs_to_forward) == 0
		assert len(block.inputs_to_keep) == 1

		assert torch.allclose(block.act_to_send[0][0], torch.full((2, 2), 2.0, device=local_rank))

		w = block.send_forward()  # Should be matched by a recv_forward
		assert w is None
		assert len(block.act_to_send) == 0

		w = block.recv_backward(2)
		assert w is None
		assert len(block.grads_to_backward) == 1

		y = block.backward()
		assert y is None  # only the last block should return its result
		assert len(block.grads_to_backward) == 0
		assert len(block.act_to_keep) == 0
		assert len(block.inputs_to_keep) == 0
		assert len(block.grads_to_send) == 1

		w = block.send_backward()
		assert w is None

	elif rank == 1:
		assert block.previous == 0
		assert block.next is None

		block.model.weight = nn.Parameter(
			torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], device=local_rank)
		)

		assert len(block.inputs_to_forward) == 0

		w = block.recv_forward(2)
		assert w is None
		assert len(block.inputs_to_forward) == 1

		assert len(block.act_to_send) == 0
		y = block.forward()[0]
		assert len(block.act_to_keep) == 1
		assert torch.allclose(y, torch.full((2, 3), 6.0, device=local_rank))
		assert y is block.act_to_keep[0][0]

		assert len(block.act_to_send) == 0  # no next block, nothing to send
		w = block.send_forward()  # does nothing
		assert w is None
		assert len(block.act_to_keep) == 1
		assert len(block.act_to_send) == 0

		w = block.recv_backward(2)  # does nothing either
		assert w is None
		assert len(block.grads_to_backward) == 0
		assert len(block.act_to_keep) == 1

		output = y.detach().requires_grad_()
		output.sum().backward()
		grads = [_fake_p2p(output.grad.data)]
		block.grads_to_backward.append(grads)

		block.backward()
		assert len(block.grads_to_backward) == 0
		assert len(block.act_to_keep) == 0
		assert len(block.inputs_to_keep) == 0
		assert len(block.grads_to_send) == 1

		w = block.send_backward()
		assert w is None
		assert len(block.grads_to_send) == 0


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

	pipe = Pipeline(model, sample, placement=[1, 2, 1, 2, 0, 3])
	if rank == 1 or rank == 2:
		assert len(pipe.blocks) == 2
	else:
		assert len(pipe.blocks) == 1

	pipe = Pipeline(model, sample, placement=[1, 2, 2, 3, 0, 3])
	if rank == 0 or rank == 1:
		assert len(pipe.blocks) == 1
	else:
		assert len(pipe.blocks) == 2

	# Test predefined partition
	pipe = Pipeline(model, sample, partition=None)
	assert len(pipe.blocks) == 1

	# Handmade placement
	pipe = Pipeline(model, sample, placement=[0, 1, 2, 3])
	if rank == 0:
		assert len(pipe.blocks) == 1
		b1 = pipe.blocks[0]
		assert b1.rank == rank
		assert b1.id == 0
		assert b1.previous is None
		assert b1.next == 1

	elif rank == 1:
		assert len(pipe.blocks) == 1
		b1 = pipe.blocks[0]
		assert b1.rank == rank
		assert b1.id == 1
		assert b1.previous == 0
		assert b1.next == 2

	# Handmade partition
	model = nn.Linear(1, 1)
	pipe = Pipeline([model], sample, partition=None)
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
	pipe = Pipeline([model], sample, placement=[0, 1, 2, 3], partition=False)
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
		assert len(b.inputs_to_forward) == 0
		assert len(b.inputs_to_keep) == 0
		assert len(b.act_to_keep) == 0
		assert len(b.act_to_send) == 0
		assert len(b.grads_to_backward) == 0

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
