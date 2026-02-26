import pytest
import torch
import torch.nn as nn

from elf.zb_utils import LinearDW
from tests.perf.zb_utils.perf_utils import (
	benchmark_operation,
	compute_statistics,
	print_performance_report,
	assert_performance,
)


@pytest.mark.perf
def test_linear_dw_performance():
	"""Compare performance of regular Linear vs decoupled LinearDW with statistical analysis"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.manual_seed(42)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(42)

	# Use larger problem sizes on GPU to amortize Python/framework overhead
	if device.type == "cuda":
		B, D_in, D_out = 512, 4096, 4096
		n_iters_per_trial = 30
		n_trials = 30
		n_warmup = 50
	else:
		B, D_in, D_out = 128, 1024, 2048
		n_iters_per_trial = 200
		n_trials = 50
		n_warmup = 100

	# Regular Linear
	linear_ref = nn.Linear(D_in, D_out).to(device)
	x_ref = torch.randn((B, D_in), requires_grad=True, device=device)

	def regular_operation():
		out = linear_ref(x_ref)
		loss = out.sum()
		loss.backward()
		linear_ref.zero_grad()
		x_ref.grad = None

	times_regular = benchmark_operation(regular_operation, n_trials, n_iters_per_trial, n_warmup)

	# Decoupled LinearDW
	linear = nn.Linear(D_in, D_out).to(device)
	linear.load_state_dict(linear_ref.state_dict())
	linear_dw = LinearDW(linear)
	torch.manual_seed(42)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(42)
	x_dw = torch.randn((B, D_in), requires_grad=True, device=device)

	def decoupled_operation():
		out = linear_dw(x_dw)
		linear_dw.move_last_computed("input", 0)
		loss = out.sum()
		loss.backward()
		linear_dw.move_last_computed("grad_output", 0)
		linear_dw.backward(0)
		linear_dw.zero_grad()
		x_dw.grad = None

	times_decoupled = benchmark_operation(decoupled_operation, n_trials, n_iters_per_trial, n_warmup)

	# Compute statistics
	stats_regular = compute_statistics(times_regular)
	stats_decoupled = compute_statistics(times_decoupled)

	# Print report
	print_performance_report(
		"Linear",
		stats_regular,
		stats_decoupled,
		device,
		n_trials,
		n_iters_per_trial,
		n_warmup,
		B=B,
		D_in=D_in,
		D_out=D_out,
	)

	# Assert performance meets thresholds
	assert_performance(stats_regular, stats_decoupled, device)
