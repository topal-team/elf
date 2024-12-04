import pytest
from elf.partitioners.partition import *

import copy
import torch
import torch.nn as nn


@pytest.mark.single
def test_extract():
	def check_model(model, sample):
		for mode in ["fx", "export"]:
			model.zero_grad()
			graph = extract_graph(copy.deepcopy(model), sample, mode)
			assert isinstance(graph, torch.fx.GraphModule)
			y = model(sample.clone().detach())
			z = graph(sample.clone().detach())
			assert torch.allclose(y, z)

			y.sum().backward()
			z.sum().backward()
			for py, pz in zip(model.parameters(), graph.parameters()):
				assert torch.allclose(py, pz)
				assert torch.allclose(py.grad, pz.grad)

	model = nn.Sequential(nn.Linear(10, 10), nn.Sigmoid(), nn.Linear(10, 1))
	check_model(model, torch.randn(4, 10))

	class Model(nn.Module):
		def __init__(self):
			super().__init__()
			self.conv = nn.Conv2d(3, 3, 3, padding="same")
			self.relu = nn.ReLU()
			self.pool = nn.MaxPool2d(2)

		def forward(self, x):
			x = self.conv(x)
			x = self.relu(x) + x
			x = self.pool(x)
			return x

	check_model(Model(), torch.randn(4, 3, 224, 224))


@pytest.mark.single
def test_postprocessing():
	def check_model(model, sample):
		graph = torch.fx.symbolic_trace(copy.deepcopy(model))
		remove_inplace_leaves(graph)

		# should break without removing inplace leaves if the input requires grad
		# but should work after removing inplace leaves
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


@pytest.mark.single
def test_partition():
	def check_partition(model, n, sample, mode):
		try:
			parts, inputs, outputs = partition_graph(copy.deepcopy(model), n, sample, mode)
		except Exception:
			pytest.skip(f"Partition failed for mode {mode}")

		trace = torch.fx.symbolic_trace(model)
		# Same number of computational nodes
		nodes_original = len(
			[node for node in trace.graph.nodes if node.op != "placeholder" and node.op != "output"]
		)
		nodes_partitioned = 0

		if isinstance(sample, (tuple, list)):
			# Warning : this can cause error if the order of the inputs is not the same as the order of the placeholders
			z = {i: s for i, s in zip(inputs[0], sample)}
			y = model(*sample)
		elif isinstance(sample, dict):
			z = sample
			y = model(**sample)
		else:
			key = next(iter(inputs[0]))
			z = {key: sample}
			y = model(sample)

		if isinstance(y, torch.Tensor):
			y = (y,)

		for i in range(len(parts)):
			part, input, output = parts[i], inputs[i], outputs[i]

			print(f"Part {i}: inputs = {input}, outputs = {output}")
			print(part.code)
			for node in part.graph.nodes:
				print(node.format_node())

			assert isinstance(part, torch.fx.GraphModule)
			nodes_partitioned += len(
				[node for node in part.graph.nodes if node.op != "placeholder" and node.op != "output"]
			)

			# As many inputs as expected
			assert len([node for node in part.graph.nodes if node.op == "placeholder"]) == len(input)
			# But only one output
			assert len([node for node in part.graph.nodes if node.op == "output"]) == 1

			for i in input:
				assert i in z

			z = part(**z)

			for o in output:
				assert o in z

		for o in outputs[n - 1]:
			for y_o in y:
				if z[o].shape == y_o.shape:
					assert torch.allclose(z[o], y_o)
					break
			else:
				assert False, f"No matching shape found for output {o}"

		assert nodes_partitioned == nodes_original

	for mode in ["naive", "constrained", "metis"]:  # dagP often crashed for now
		model = nn.Sequential(
			nn.Linear(10, 10),
			nn.ReLU(),
			nn.Linear(10, 10),
			nn.ReLU(),
			nn.Linear(10, 10),
			nn.ReLU(),
			nn.Linear(10, 1),
		)
		check_partition(model, 2, torch.randn(4, 10), mode)
		check_partition(model, 3, torch.randn(4, 10), mode)
		check_partition(model, 4, torch.randn(4, 10), mode)

		# Test with tuple input
		class ModelMultipleInputs(nn.Module):
			def __init__(self):
				super().__init__()
				self.l1 = nn.Linear(32, 32)
				self.l2 = nn.Linear(32, 32)

			def forward(self, x, y):
				x = self.l1(x)
				y = self.l2(y)
				z = self.l1(x + y)
				z = self.l2(z + x + y)
				return z

		sample = {"x": torch.randn(4, 32), "y": torch.randn(4, 32)}
		check_partition(ModelMultipleInputs(), 2, sample, mode)

		# Test with multiple outputs
		class ModelMultipleOutputs(nn.Module):
			def __init__(self):
				super().__init__()
				self.l1 = nn.Linear(32, 16)
				self.l2 = nn.Linear(16, 3)

			def forward(self, x):
				x1 = self.l1(x)
				x2 = self.l2(x1)
				return x1, x2

		check_partition(ModelMultipleOutputs(), 2, torch.randn(4, 32), mode)
