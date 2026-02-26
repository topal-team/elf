import pytest

import torch.nn as nn

from elf.zb_utils import (
	Conv1dDW,
	LinearDW,
	replace_linear_with_linear_dw,
	replace_layer_with_layer_dw,
)


@pytest.mark.unit
def test_replace_linear_with_linear_dw():
	model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10), nn.Conv2d(10, 10, 3, 1, 1))

	ptrs = (model[0].weight, model[0].bias, model[2].weight, model[2].bias)

	assert isinstance(model[0], nn.Linear)
	assert isinstance(model[2], nn.Linear)
	assert isinstance(model[3], nn.Conv2d)

	replace_linear_with_linear_dw(model, "cpu")

	assert isinstance(model[0], LinearDW)
	assert isinstance(model[1], nn.ReLU)
	assert isinstance(model[2], LinearDW)
	assert isinstance(model[3], nn.Conv2d)

	# Check that parameters were not copied, but reused
	new_ptrs = (model[0].weight, model[0].bias, model[2].weight, model[2].bias)
	for ptr, new_ptr in zip(ptrs, new_ptrs):
		assert ptr is new_ptr


@pytest.mark.unit
def test_replace_layer_with_layer_dw():
	model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Conv1d(10, 10, 3, 1, 1))

	ptrs = (model[0].weight, model[0].bias, model[2].weight, model[2].bias)

	assert isinstance(model[0], nn.Linear)
	assert isinstance(model[1], nn.ReLU)
	assert isinstance(model[2], nn.Conv1d)

	replace_layer_with_layer_dw(model)

	assert isinstance(model[0], LinearDW)
	assert isinstance(model[1], nn.ReLU)
	assert isinstance(model[2], Conv1dDW)

	new_ptrs = (model[0].weight, model[0].bias, model[2].weight, model[2].bias)
	for ptr, new_ptr in zip(ptrs, new_ptrs):
		assert ptr is new_ptr

	model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Conv1d(10, 10, 3, 1, 1))
	replace_layer_with_layer_dw(model, only=[nn.Linear])

	assert isinstance(model[0], LinearDW)
	assert isinstance(model[1], nn.ReLU)
	assert isinstance(model[2], nn.Conv1d)
