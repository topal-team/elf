import copy
import pytest

import torch
import torch.nn as nn

from elf.registry import TRACERS


@pytest.mark.unit
def test_extract():
	def check_model(model, sample):
		for tracer in TRACERS.available():
			model.zero_grad()
			graph = TRACERS[tracer](copy.deepcopy(model), sample)
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
