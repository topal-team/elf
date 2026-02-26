import pytest
import torch
import torch.nn as nn

from elf.zb_utils import Conv1dDW
from tests.perf.zb_utils.perf_utils import (
	benchmark_operation,
	compute_statistics,
	print_performance_report,
	assert_performance,
)


@pytest.mark.perf
def test_conv1d_dw_performance():
	"""Compare performance of regular Conv1d vs decoupled Conv1dDW with statistical analysis"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(42)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(42)

	# Use larger problem sizes on GPU to amortize Python/framework overhead
	if device.type == "cuda":
		B, C_in, C_out, L = 64, 256, 512, 1024
		n_iters_per_trial = 30
		n_trials = 30
		n_warmup = 50
	else:
		B, C_in, C_out, L = 32, 64, 128, 256
		n_iters_per_trial = 200
		n_trials = 50
		n_warmup = 100

	kernel_size, stride, padding = 3, 1, 1

	# Regular Conv1d
	conv_ref = nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding).to(device)
	x_ref = torch.randn((B, C_in, L), requires_grad=True, device=device)

	def regular_operation():
		out = conv_ref(x_ref)
		loss = out.sum()
		loss.backward()
		conv_ref.zero_grad()
		x_ref.grad = None

	times_regular = benchmark_operation(regular_operation, n_trials, n_iters_per_trial, n_warmup)

	# Decoupled Conv1dDW
	conv = nn.Conv1d(C_in, C_out, kernel_size, stride=stride, padding=padding).to(device)
	conv.load_state_dict(conv_ref.state_dict())
	conv_dw = Conv1dDW(conv)
	x_dw = torch.randn((B, C_in, L), requires_grad=True, device=device)
	torch.manual_seed(42)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(42)
	x_dw = torch.randn((B, C_in, L), requires_grad=True, device=device)

	def decoupled_operation():
		out = conv_dw(x_dw)
		conv_dw.move_last_computed("input", 0)
		loss = out.sum()
		loss.backward()
		conv_dw.move_last_computed("grad_output", 0)
		conv_dw.backward(0)
		conv_dw.zero_grad()
		x_dw.grad = None

	times_decoupled = benchmark_operation(decoupled_operation, n_trials, n_iters_per_trial, n_warmup)

	# Compute statistics
	stats_regular = compute_statistics(times_regular)
	stats_decoupled = compute_statistics(times_decoupled)

	# Print report
	print_performance_report(
		"Conv1d",
		stats_regular,
		stats_decoupled,
		device,
		n_trials,
		n_iters_per_trial,
		n_warmup,
		B=B,
		C_in=C_in,
		C_out=C_out,
		L=L,
		kernel=kernel_size,
	)

	# Assert performance meets thresholds
	assert_performance(stats_regular, stats_decoupled, device)
