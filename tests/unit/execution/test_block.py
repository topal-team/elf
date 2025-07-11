import pytest

import torch
import torch.nn as nn

from elf.execution.block import PipelineBlock, Variable
from elf.utils import TensorMetadata
from elf.partitioners.utils import Signature


@pytest.mark.unit
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


@pytest.mark.unit
def test_variable_wait_and_pop():
	var = Variable("test", peer=1, group=None)
	tensor = torch.randn(2, 3)
	var.to_process.append((None, tensor))

	result = var.wait_and_pop(0)
	assert torch.equal(result, tensor)
	assert all(value is None for value in var.to_process)


@pytest.mark.unit
def test_variable_get_buffer():
	var = Variable("test", peer=1, group=None)
	tensor = torch.randn(2, 3)
	var.metadata = TensorMetadata(tensor)

	buffer = var.get_buffer(4)
	assert buffer.shape == (4, 2, 3)
	assert buffer.dtype == tensor.dtype


@pytest.mark.unit
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
	assert len(block.input_variables) == 1
	assert len(block.output_variables) == 1
	assert block.input_variables[0].name == "input"
	assert block.output_variables[0][0].name == "output"


@pytest.mark.unit
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
	block.input_variables[0].set(block.input_variables[0].to_process, 0, (None, input_tensor))
	block.forward(0)

	expected = torch.tensor([[5.1, 11.2]], device=device)
	actual = block.output_variables[0][0].get(block.output_variables[0][0].to_send, 0)
	assert torch.allclose(actual, expected)


@pytest.mark.unit
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
	block.input_variables[0].set(block.input_variables[0].to_process, 0, (None, input_tensor))
	block.forward(0)

	# Backward pass
	grad_tensor = torch.tensor([[1.0, 1.0]], device=device)
	block.output_variables[0][0].set(block.output_variables[0][0].to_process, 0, (None, grad_tensor))
	block.backward_inputs(0)
	block.backward_params(0)

	expected_input_grad = torch.tensor([[4.0, 6.0]], device=device)
	actual_input_grad = block.input_variables[0].get(block.input_variables[0].to_send, 0)
	assert torch.allclose(actual_input_grad, expected_input_grad)

	expected_weight_grad = torch.tensor([[1.0, 2.0], [1.0, 2.0]], device=device)
	assert torch.allclose(model.weight.grad, expected_weight_grad)
