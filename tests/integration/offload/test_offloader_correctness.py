import pytest

import torch
import torch.nn as nn

from elf.execution.offload import OffloadToCPU
from models.simple import SimpleFFN, SimpleCNN, SimpleTransformer, SimpleResNet


def _all_close(a, b):
	return torch.allclose(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize("model_name", ["ffn", "cnn", "transformer", "resnet"])
def test_offload_forward_backward_matches(model_name):
	device = torch.device("cuda")
	torch.manual_seed(0)
	torch.cuda.manual_seed_all(0)

	# Create model based on parameter
	with torch.device(device):
		if model_name == "ffn":
			model_no = SimpleFFN(32, 128, 3, nn.Identity)
			model_off = SimpleFFN(32, 128, 3, nn.Identity)
		elif model_name == "cnn":
			model_no = SimpleCNN(128)
			model_off = SimpleCNN(128)
		elif model_name == "transformer":
			model_no = SimpleTransformer(128, 64, 3, seq_len=32)
			model_off = SimpleTransformer(128, 64, 3, seq_len=32)
		elif model_name == "resnet":
			model_no = SimpleResNet(3, 256, 10)
			model_off = SimpleResNet(3, 256, 10)

	model_off.load_state_dict(model_no.state_dict())

	model_off.eval()
	model_no.eval()

	x = model_no.get_sample(16, device=device)
	x2 = x.detach().clone()

	# Baseline (no offload)
	out_no = model_no(x)
	loss_no = out_no.float().sum()
	loss_no.backward()
	grads_no = {n: p.grad.detach().clone() for n, p in model_no.named_parameters()}

	# With offload
	model_off.zero_grad(set_to_none=True)
	x2.grad = None
	with OffloadToCPU() as off:
		out_off = model_off(x2)

	loss_off = out_off.float().sum()
	off.prefetch()
	loss_off.backward()
	off.release()

	torch.cuda.synchronize()

	grads_off = {n: p.grad.detach().clone() for n, p in model_off.named_parameters()}

	assert _all_close(out_off, out_no), f"Output mismatch, diff: {torch.norm(out_off - out_no)}"
	assert _all_close(loss_off, loss_no), f"Loss mismatch, diff: {torch.norm(loss_off - loss_no)}"
	for n in grads_no:
		assert _all_close(grads_off[n], grads_no[n]), (
			f"Grad mismatch for {n}, diff: {torch.norm(grads_off[n] - grads_no[n])}"
		)


class BigMLP(nn.Module):
	def __init__(self, width: int = 2048, depth: int = 6):
		super().__init__()
		layers = []
		for i in range(depth):
			layers.append(nn.Linear(width, width))
			layers.append(nn.ReLU())
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def _sync():
	if torch.cuda.is_available():
		torch.cuda.synchronize()


@pytest.mark.integration
@pytest.mark.gpu
def test_offload_reduces_gpu_memory_and_prefetch_increases_temporarily():
	device = torch.device("cuda")
	torch.manual_seed(0)
	torch.cuda.manual_seed_all(0)

	model = BigMLP(width=2048, depth=6).to(device)
	batch = 1024
	x = torch.randn(batch, 2048, device=device, requires_grad=True)

	# Warm-up to stabilize allocator
	_ = model(x).sum().backward()
	model.zero_grad(set_to_none=True)
	x.grad = None
	_sync()
	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()

	# No offload: measure forward resident memory (after model and input are already on device)
	_sync()
	base_before = torch.cuda.memory_allocated()
	_ = model(x).sum()
	_sync()
	no_offload_after_fwd = torch.cuda.memory_allocated()

	# Clean up graph to release activations
	model.zero_grad(set_to_none=True)
	x.grad = None

	# With offload: measure forward resident memory
	torch.cuda.empty_cache()
	torch.cuda.reset_peak_memory_stats()
	_sync()
	with OffloadToCPU() as off:
		_sync()
		base_before_off = torch.cuda.memory_allocated()
		_ = model(x).sum()
		_sync()
		offload_after_fwd = torch.cuda.memory_allocated()
		# Prefetch should allocate GPU buffers, increasing memory
		off.prefetch()
		_sync()
		after_prefetch = torch.cuda.memory_allocated()
		# Finish backward to clear graphs
		model.zero_grad(set_to_none=True)
		x.grad = None
		(_ := model(x).sum()).backward()
		_sync()
		off.release()

	# Assertions
	no_offload_delta = no_offload_after_fwd - base_before
	offload_delta = offload_after_fwd - base_before_off

	# Offloading should reduce resident GPU memory during forward
	assert offload_delta < no_offload_delta, (
		f"Expected offload to reduce forward GPU memory: offload={offload_delta}, no_offload={no_offload_delta}"
	)

	# Prefetch should increase memory vs immediately after forward with offload
	assert after_prefetch >= offload_after_fwd, (
		f"Prefetch should increase or equal GPU memory: prefetch={after_prefetch}, fwd={offload_after_fwd}"
	)
