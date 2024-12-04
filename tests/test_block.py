import os
from datetime import timedelta

import pytest

from elf.block import PipelineBlock, Variable
from elf.utils import TensorMetadata
from elf.partitioners.utils import Signature

import torch
import torch.nn as nn
import torch.distributed as dist


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


@pytest.mark.single
def test_variable_init():
	var = Variable("test", peer=1, group=None)
	assert var.name == "test"
	assert var.peer == 1
	assert var.group is None
	assert var.metadata is None
	assert var.was_metadata_sent is False
	assert isinstance(var.to_process, list)
	assert isinstance(var.saved, list)
	assert isinstance(var.to_send, list)


@pytest.mark.single
def test_variable_wait_and_pop():
	var = Variable("test", peer=1, group=None)
	tensor = torch.randn(2, 3)
	var.to_process.append((None, tensor))

	result = var.wait_and_pop(0)
	assert torch.equal(result, tensor)
	assert all(value is None for value in var.to_process)


@pytest.mark.single
def test_variable_get_buffer():
	var = Variable("test", peer=1, group=None)
	tensor = torch.randn(2, 3)
	var.metadata = TensorMetadata(tensor)

	buffer = var.get_buffer(4)
	assert buffer.shape == (4, 2, 3)
	assert buffer.dtype == tensor.dtype


@pytest.mark.single
def test_pipeline_block_init():
	model = nn.Linear(10, 5)
	placement = [0, 1, 2]
	signature = Signature(inputs=["input"], outputs=["output"], sources=[None], targets=[[1]])

	block = PipelineBlock(
		model=model, id_=0, placement=placement, signature=signature, pp_group=None, dp_group=None
	)

	assert block.id == 0
	assert block.rank == 0
	assert block.is_first
	assert not block.is_last
	assert len(block.inputs) == 1
	assert len(block.outputs) == 1
	assert block.inputs[0].name == "input"
	assert block.outputs[0][0].name == "output"


@pytest.mark.single
def test_pipeline_block_forward():
	model = nn.Linear(2, 2)
	model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
	model.bias.data = torch.tensor([0.1, 0.2])

	device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

	placement = [0, 1]
	signature = Signature(inputs=["input"], outputs=["output"], sources=[None], targets=[[1]])

	block = PipelineBlock(
		model=model, id_=0, placement=placement, signature=signature, pp_group=None, dp_group=None
	)

	input_tensor = torch.tensor([[1.0, 2.0]], device=device)
	block.inputs[0].set(block.inputs[0].to_process, 0, (None, input_tensor))
	block.forward(0)

	expected = torch.tensor([[5.1, 11.2]], device=device)
	actual = block.outputs[0][0].get(block.outputs[0][0].to_send, 0)
	assert torch.allclose(actual, expected)


@pytest.mark.single
def test_pipeline_block_backward():
	model = nn.Linear(2, 2)
	model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
	model.bias.data = torch.tensor([0.1, 0.2])

	device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

	placement = [0, 1]
	signature = Signature(inputs=["input"], outputs=["output"], sources=[None], targets=[[1]])

	block = PipelineBlock(
		model=model, id_=0, placement=placement, signature=signature, pp_group=None, dp_group=None
	)

	# Forward pass first
	input_tensor = torch.tensor([[1.0, 2.0]], requires_grad=True, device=device)
	block.inputs[0].set(block.inputs[0].to_process, 0, (None, input_tensor))
	block.forward(0)

	# Backward pass
	grad_tensor = torch.tensor([[1.0, 1.0]], device=device)
	block.outputs[0][0].set(block.outputs[0][0].to_process, 0, (None, grad_tensor))
	block.backward(0)

	# Check input gradients
	expected_input_grad = torch.tensor([[4.0, 6.0]], device=device)
	actual_input_grad = block.inputs[0].get(block.inputs[0].to_send, 0)
	assert torch.allclose(actual_input_grad, expected_input_grad)

	# Check weight gradients
	expected_weight_grad = torch.tensor([[1.0, 2.0], [1.0, 2.0]], device=device)
	assert torch.allclose(model.weight.grad, expected_weight_grad)


@pytest.mark.multi
def test_block_communication(init_dist):
	rank, local_rank, world_size = init_dist

	if world_size < 2:
		pytest.skip("This test needs at least 2 processes to run")
		return

	# Create two connected blocks on different ranks
	if rank == 0:
		model = nn.Linear(2, 2)
		model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=local_rank)
		model.bias.data = torch.tensor([0.1, 0.2], device=local_rank)

		signature = Signature(inputs=["input"], outputs=["output"], sources=[None], targets=[[1]])

		block = PipelineBlock(
			model=model, id_=0, placement=[0, 1], signature=signature, pp_group=None, dp_group=None
		)

		# Forward pass
		input_tensor = torch.tensor([[1.0, 2.0]], device=local_rank)
		block.inputs[0].set(block.inputs[0].to_process, 0, (None, input_tensor))
		block.forward(0)

		# Send output to rank 1
		block.send_forward(0, dst=1)

		# Receive gradients from rank 1 for backward
		block.recv_backward(0, mb_size=1, src=1)
		block.backward(0)

	elif rank == 1:
		model = nn.Linear(2, 2)
		model.weight.data = torch.tensor([[0.5, 0.6], [0.7, 0.8]], device=local_rank)
		model.bias.data = torch.tensor([0.3, 0.4], device=local_rank)

		signature = Signature(inputs=["input"], outputs=["output"], sources=[0], targets=[[None]])

		block = PipelineBlock(
			model=model, id_=1, placement=[0, 1], signature=signature, pp_group=None, dp_group=None
		)

		# Receive input from rank 0
		block.recv_forward(0, mb_size=1, src=0)
		block.forward(0)

		# Create gradients and send back
		grad_tensor = torch.tensor([[1.0, 1.0]], device=local_rank)
		block.outputs[0][0].set(block.outputs[0][0].to_process, 0, (None, grad_tensor))
		block.backward(0)
		block.send_backward(0, dst=0)

	dist.barrier()  # Ensure both processes complete
