import pytest
import torch
import torch.nn as nn

from elf.zb_utils import LinearDW


@pytest.mark.integration
@pytest.mark.gpu
def test_layerdw_memory_lifecycle():
	if not torch.cuda.is_available():
		pytest.skip("CUDA not available")

	def close(a, b):
		tolerance = 256 * 1024  # 256 KB
		return abs(a - b) <= tolerance

	def assert_close(a, b):
		tolerance = 512 * 1024  # 512 KB
		assert abs(a - b) <= tolerance, (
			f"Memory difference too high: expected {b / 1024 / 1024:.2f} MB, got {a / 1024 / 1024:.2f} MB"
		)

	def assert_less(a, b):
		assert a <= b, f"Expected {a / 1024 / 1024:.2f} MB <= {b / 1024 / 1024:.2f} MB"

	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()

	device = torch.device("cuda")

	model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)).to(device)

	zb_model = nn.Sequential(
		LinearDW(nn.Linear(256, 256)), nn.ReLU(), LinearDW(nn.Linear(256, 256))
	).to(device)

	x = torch.randn(1024, 256, device=device)

	# Warmup to initialize cuda contexts etc
	model(x).sum().backward()
	zb_model(x).sum().backward()

	mem_before = torch.cuda.memory_allocated()

	# Forward
	y = model(x)
	torch.cuda.synchronize()
	mem_after_fwd = torch.cuda.memory_allocated()

	# Backward
	y.mean().backward()
	torch.cuda.synchronize()
	mem_after_autograd = torch.cuda.memory_allocated()

	del y

	assert_close(torch.cuda.memory_allocated(), mem_before)

	# Forward
	y = zb_model(x)
	torch.cuda.synchronize()
	assert_less(torch.cuda.memory_allocated(), mem_after_fwd)

	# Backward Inputs
	y.mean().backward()
	for layer in zb_model:
		if isinstance(layer, LinearDW):
			layer.move_last_computed("input", 0)
			layer.move_last_computed("grad_output", 0)

	# Some tensors are still in memory
	torch.cuda.synchronize()
	assert_less(mem_after_autograd, torch.cuda.memory_allocated())

	# Decoupled param backward frees stored tensors
	for layer in zb_model:
		if isinstance(layer, LinearDW):
			layer.backward(0)

	torch.cuda.synchronize()

	# Memory should be roughly back to forward level
	assert_less(torch.cuda.memory_allocated(), mem_after_fwd)
