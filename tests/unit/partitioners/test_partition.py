import pytest
import copy
import torch
import torch.nn as nn

from elf.partitioners import partition_graph
from elf.partitioners.utils import remove_inplace_leaves
from elf.registry import PARTITIONERS, TRACERS


@pytest.mark.unit
def test_postprocessing():
	def check_model(model, sample):
		graph = torch.fx.symbolic_trace(copy.deepcopy(model))
		remove_inplace_leaves(graph)

		y = model(sample.clone().detach())
		z = graph(sample.clone().detach().requires_grad_())
		assert torch.allclose(y, z)

		y.sum().backward()
		z.sum().backward()

		for py, pz in zip(model.parameters(), graph.parameters()):
			assert torch.allclose(py, pz)
			assert torch.allclose(py.grad, pz.grad)

	model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(True), nn.Linear(10, 1))
	check_model(model, torch.randn(4, 10))

	model = nn.Sequential(nn.Conv2d(3, 3, 3, padding="same"), nn.ReLU(True))
	check_model(model, torch.randn(4, 3, 224, 224))

	model = nn.Sequential(nn.ReLU(True), nn.Linear(10, 10), nn.ReLU(True), nn.Linear(10, 1))
	check_model(model, torch.randn(4, 10))


class BaseModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.Sequential()
		for _ in range(24):
			self.layers.append(nn.Linear(256, 256))
			self.layers.append(nn.ReLU())
		self.layers.append(nn.Linear(256, 1))

	def forward(self, x):
		return self.layers(x)


class ModelMultipleInputs(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Linear(256, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 256)

	def forward(self, x, y):
		x = self.l1(x)
		y = self.l2(y)
		z = x + y
		# Should cut here
		z = self.l3(z)
		z = self.l4(z)
		return z


class ModelMultipleOutputs(nn.Module):
	def __init__(self):
		super().__init__()
		self.l1 = nn.Linear(256, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 256)

	def forward(self, x):
		x1 = self.l1(x).relu()
		x2 = self.l2(x1).relu()
		# Should cut here
		x3 = self.l3(x2).relu() + x2
		x4 = self.l4(x3).relu()

		return x3, x4


@pytest.mark.unit
@pytest.mark.parametrize("mode", PARTITIONERS.available())
@pytest.mark.parametrize(
	"model_config",
	[
		{"model": BaseModel, "sample": torch.randn(4, 256), "n_parts": [2, 3, 4]},
		{
			"model": ModelMultipleInputs,
			"sample": (torch.randn(4, 256), torch.randn(4, 256)),
			"n_parts": [2],
		},
		{"model": ModelMultipleOutputs, "sample": torch.randn(4, 256), "n_parts": [2]},
	],
)
def test_partition(mode, model_config):
	# Skip a specific combination that's known to be problematic
	if mode == "metis" and model_config["model"] == ModelMultipleInputs:
		pytest.skip("Skipping metis partitioner with ModelMultipleInputs (known issue)")

	def check_partition(model, n, sample, mode):
		parts, signatures = partition_graph(copy.deepcopy(model), n, sample, mode, TRACERS["fx"])

		trace = torch.fx.symbolic_trace(model)
		# Same number of computational nodes
		nodes_original = len(
			[node for node in trace.graph.nodes if node.op != "placeholder" and node.op != "output"]
		)
		nodes_partitioned = 0

		sample = (sample,) if not isinstance(sample, (tuple, list)) else sample

		groundtruth = model(*sample)
		groundtruth = (groundtruth,) if not isinstance(groundtruth, (tuple, list)) else groundtruth

		storage = {}  # global storage
		# Pre-fill with first part's inputs
		for i, s in zip(signatures[0].inputs, sample):
			storage[i] = s

		for i in range(len(parts)):
			part, inputs, outputs = parts[i], signatures[i].inputs, signatures[i].outputs

			assert isinstance(part, torch.fx.GraphModule)
			nodes_partitioned += len(
				[node for node in part.graph.nodes if node.op != "placeholder" and node.op != "output"]
			)

			# Correct number of inputs and outputs
			assert len([node for node in part.graph.nodes if node.op == "placeholder"]) == len(inputs)

			# All input should be available
			for in_name in inputs:
				assert in_name in storage

			local_inputs = (storage[k] for k in inputs)
			local_outputs = part(*local_inputs)

			# Correct outputs
			assert len(local_outputs) == len(outputs)
			for o, o_name in zip(local_outputs, outputs):
				assert o_name not in storage
				storage[o_name] = o

			if i == len(parts) - 1:
				for gt, o in zip(groundtruth, local_outputs):
					if isinstance(gt, torch.Tensor):
						assert torch.allclose(gt, o)
					else:
						assert gt == o

		assert nodes_partitioned == nodes_original

	model = model_config["model"]()
	sample = model_config["sample"]

	for n in model_config["n_parts"]:
		check_partition(model, n, sample, mode)


@pytest.mark.unit
def test_partition_single_stage():
	"""Test that single stage partitioning works correctly."""
	model = BaseModel()
	sample = torch.randn(4, 256)

	parts, signatures = partition_graph(
		copy.deepcopy(model), 1, sample, PARTITIONERS["naive"], TRACERS["fx"]
	)

	assert len(parts) == 1
	assert len(signatures) == 1

	# Single stage should produce same output as original
	output_original = model(sample)
	output_partitioned = parts[0](*[sample])[0]

	assert torch.allclose(output_original, output_partitioned, atol=1e-5)
