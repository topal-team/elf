import pytest
import torch
import torch.nn as nn
import itertools

from elf.zb_utils import LinearDW


@pytest.mark.unit
@pytest.mark.parametrize(
	"bias,in_features,out_features",
	list(
		itertools.product(
			[True, False],  # bias
			[8, 64, 1024],  # in_features
			[8, 64, 1024],  # out_features
		)
	),
)
def test_linear_dw_match_linear(bias, in_features, out_features):
	torch.manual_seed(0)
	x = torch.randn(4, in_features, requires_grad=True)

	# Reference linear
	ref = nn.Linear(in_features, out_features, bias=bias)
	ref_out = ref(x)
	loss_ref = ref_out.sum()
	loss_ref.backward()
	grad_w_ref = ref.weight.grad.clone()
	grad_b_ref = ref.bias.grad.clone() if bias else None

	# LinearDW wrapper around a copy of ref
	lin_copy = nn.Linear(in_features, out_features, bias=bias)
	lin_copy.load_state_dict(ref.state_dict())
	ldw = LinearDW(lin_copy, device=torch.device("cpu"))

	x2 = x.clone().detach().requires_grad_(True)
	y = ldw(x2)
	assert torch.allclose(y, ref_out)

	ldw.move_last_computed("input", 0)
	loss = y.sum()
	loss.backward()
	assert torch.allclose(x2.grad, x.grad)
	ldw.move_last_computed("grad_output", 0)

	# Now compute parameter grads via decoupled backward
	ldw.backward(0)

	assert torch.allclose(ldw.weight.grad, grad_w_ref, atol=1e-6)
	if ldw.bias is not None:
		assert torch.allclose(ldw.bias.grad, grad_b_ref, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize(
	"bias,in_features,out_features",
	list(
		itertools.product(
			[True, False],  # bias
			[8, 64],  # in_features
			[8, 64],  # out_features
		)
	),
)
def test_linear_dw_match_linear_complex(bias, in_features, out_features):
	"""Test LinearDW with complex tensors (requires conjugate transpose for weight gradients)"""
	torch.manual_seed(0)
	# Create complex input
	x = torch.randn(4, in_features, dtype=torch.complex64, requires_grad=True)

	# Reference linear with complex weights
	ref = nn.Linear(in_features, out_features, bias=bias, dtype=torch.complex64)
	ref_out = ref(x)
	loss_ref = ref_out.abs().sum()  # Use abs().sum() as loss for complex tensors
	loss_ref.backward()
	grad_w_ref = ref.weight.grad.clone()
	grad_b_ref = ref.bias.grad.clone() if bias else None

	# LinearDW wrapper around a copy of ref
	lin_copy = nn.Linear(in_features, out_features, bias=bias, dtype=torch.complex64)
	lin_copy.load_state_dict(ref.state_dict())
	ldw = LinearDW(lin_copy, device=torch.device("cpu"))

	x2 = x.clone().detach().requires_grad_(True)
	y = ldw(x2)
	assert torch.allclose(y, ref_out)

	ldw.move_last_computed("input", 0)
	loss = y.abs().sum()
	loss.backward()

	# Check input gradients
	assert torch.allclose(x2.grad, x.grad, atol=1e-5), (
		f"Complex input gradient mismatch: max diff = {(x2.grad - x.grad).abs().max()}\n"
		f"LinearDX.backward must use weight.conj() for complex tensors"
	)
	ldw.move_last_computed("grad_output", 0)

	ldw.backward(0)

	assert torch.allclose(ldw.weight.grad, grad_w_ref, atol=1e-5), (
		f"Complex weight gradient mismatch: max diff = {(ldw.weight.grad - grad_w_ref).abs().max()}"
	)
	if ldw.bias is not None:
		assert torch.allclose(ldw.bias.grad, grad_b_ref, atol=1e-5), (
			f"Complex bias gradient mismatch: max diff = {(ldw.bias.grad - grad_b_ref).abs().max()}"
		)
