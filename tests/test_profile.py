from elf.partitioners.profile import *
import pytest

import copy
import psutil

import torch
import torch.nn as nn


def get_current_memory():
	if torch.cuda.is_available():
		return torch.cuda.memory_allocated()
	else:
		return psutil.Process().memory_info().rss


@pytest.mark.single
def test_profile():
	def check_model(model, sample):
		trace = torch.fx.symbolic_trace(copy.deepcopy(model))

		memory_before = get_current_memory()

		times, memories = profile_operations(trace, sample)

		memory_after = get_current_memory()
		assert (
			memory_after - memory_before < 100e6
		), f"Memory leak detected: Before {memory_before}, After {memory_after}, diff = {memory_after - memory_before}"

		assert len(times) == len(trace.graph.nodes)
		assert len(memories) == len(trace.graph.nodes)

		for node in trace.graph.nodes:
			assert node.name in times
			assert node.name in memories

			if node.op == "placeholder" or node.op == "output":
				assert times[node.name] == 0

		# Check that all parameters and buffers of the trace are the same as the model
		for trace_param, model_param in zip(trace.parameters(), model.parameters()):
			assert torch.allclose(trace_param, model_param)

		for trace_buffer, model_buffer in zip(trace.buffers(), model.buffers()):
			assert torch.allclose(trace_buffer, model_buffer)

		y = model(sample.clone().detach())
		z = trace(sample.clone().detach())
		assert torch.allclose(y, z)

		y.sum().backward()
		z.sum().backward()
		for trace_param, model_param in zip(trace.parameters(), model.parameters()):
			assert torch.allclose(trace_param.grad, model_param.grad)

	# Basic linear model
	model = nn.Sequential(nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 20))
	check_model(model, torch.randn(4, 100))

	# Model with buffers
	model = nn.Sequential(
		nn.Conv2d(1, 3, 3, padding="same"),
		nn.BatchNorm2d(3),
		nn.ReLU(),
		nn.MaxPool2d(2),
		nn.Flatten(),
		nn.Linear(3 * 14 * 14, 10),
	)
	check_model(model, torch.randn(4, 1, 28, 28))
