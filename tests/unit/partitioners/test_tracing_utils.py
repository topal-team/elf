import pytest
import torch
import torch.nn as nn

from elf.partitioners.tracing import (
	filter_paths,
	get_shapes,
	path_to_module,
	extract_module_by_name,
	replace_module_inplace,
)


class DummyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.block = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
		self.head = nn.Linear(4, 1)

	def forward(self, x):
		x = self.block(x)
		return self.head(x)


@pytest.mark.unit
def test_filter_paths():
	paths = ["block", "block.0", "block.0.weight", "head", "head.weight"]
	# Expect only the deepest unique leaves (no parent that is prefix of another)
	expected = {"block.0.weight", "head.weight"}
	assert filter_paths(paths) == expected


@pytest.mark.unit
def test_get_shapes():
	x = torch.randn(2, 3)
	assert get_shapes(x) == (2, 3)

	lst = [x, x]
	assert get_shapes(lst) == [(2, 3), (2, 3)]

	dct = {"a": x, "b": [x, x]}
	assert get_shapes(dct) == {"a": (2, 3), "b": [(2, 3), (2, 3)]}


@pytest.mark.unit
def test_path_to_module_and_extract():
	model = DummyModel()

	path = path_to_module(model, model.block)
	assert path == "block"

	path2 = path_to_module(model.block, model.block[0])
	assert path2 == "0"

	# extract_module_by_name uses dot notation
	extracted = extract_module_by_name(model, "block.0")
	assert isinstance(extracted, nn.Linear)


@pytest.mark.unit
def test_replace_module_inplace():
	model = DummyModel()
	new_linear = nn.Linear(4, 4)
	old_module = replace_module_inplace(model, "block.0", new_linear)

	# Returned module should be original one
	assert isinstance(old_module, nn.Linear)
	# The model should now contain the new module
	assert model.block[0] is new_linear
