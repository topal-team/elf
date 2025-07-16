import pytest
import torch
import torch.nn as nn

from elf.zb_utils import LinearDW


@pytest.mark.unit
def test_linear_dw_match_linear():
	torch.manual_seed(0)
	x = torch.randn(4, 8, requires_grad=True)

	# Reference linear
	ref = nn.Linear(8, 8, bias=True)
	ref_out = ref(x)
	loss_ref = ref_out.sum()
	loss_ref.backward()
	grad_w_ref = ref.weight.grad.clone()
	grad_b_ref = ref.bias.grad.clone()

	# LinearDW wrapper around a copy of ref
	lin_copy = nn.Linear(8, 8, bias=True)
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
