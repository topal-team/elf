import pytest
import torch
import torch.nn as nn
import itertools

from elf.zb_utils import Conv1dDW


@pytest.mark.unit
@pytest.mark.parametrize(
	"bias,padding,stride,kernel_size,dilation,groups",
	list(
		itertools.product(
			[True, False],  # bias
			["same", 0, 1, 2],  # padding: 'same' and numbers
			[1, 2, 3],  # stride
			[1, 3, 5],  # kernel size
			[1, 2, 3],  # dilation
			[1, 2],  # groups
		)
	),
)
def test_conv_dw_match_conv_1d(bias, padding, stride, kernel_size, dilation, groups):
	B, C_in, L = 4, 8, 16

	# For valid (non-scalar) padding
	if padding == "same":
		pad = "same"
	else:
		pad = int(padding)

	# Skip invalid combination: padding='same' is not supported for strided convolutions
	if padding == "same" and stride != 1:
		pytest.skip("padding='same' is not supported for strided convolutions")

	# Skip if C_in or C_out not divisible by groups
	C_out = 12
	if C_in % groups != 0 or C_out % groups != 0:
		pytest.skip(f"C_in={C_in} or C_out={C_out} not divisible by groups={groups}")

	# To avoid output size being zero when kernel/dilation/stride is high
	# Make sure input is long enough
	input_len = L + 20 if kernel_size * dilation > L else L
	x = torch.randn((B, C_in, input_len), requires_grad=True)

	ref = nn.Conv1d(
		C_in,
		C_out,
		kernel_size,
		bias=bias,
		padding=pad,
		stride=stride,
		dilation=dilation,
		groups=groups,
	)
	ref_out = ref(x)
	loss_ref = ref_out.sum()
	loss_ref.backward()
	grad_w_ref = ref.weight.grad.clone()
	grad_b_ref = ref.bias.grad.clone() if bias else None
	ref.zero_grad()

	conv1d = nn.Conv1d(
		C_in,
		C_out,
		kernel_size,
		bias=bias,
		padding=pad,
		stride=stride,
		dilation=dilation,
		groups=groups,
	)
	conv1d.load_state_dict(ref.state_dict())
	conv1dw = Conv1dDW(conv1d, device=torch.device("cpu"))

	x2 = x.clone().detach().requires_grad_(True)
	y = conv1dw(x2)
	assert torch.allclose(y, ref_out, atol=1e-6, rtol=1e-4), (
		f"Failed on fwd: bias={bias}, pad={pad}, stride={stride}, ks={kernel_size}, dil={dilation}, groups={groups}"
	)

	conv1dw.move_last_computed("input", 0)
	loss = y.sum()
	loss.backward()
	assert torch.allclose(x2.grad, x.grad, atol=1e-6, rtol=1e-4), (
		f"Failed on input grad: bias={bias}, pad={pad}, stride={stride}, ks={kernel_size}, dil={dilation}, groups={groups}"
	)
	conv1dw.move_last_computed("grad_output", 0)

	# Now compute parameter grads via decoupled backward
	conv1dw.backward(0)

	assert torch.allclose(conv1dw.weight.grad, grad_w_ref, atol=1e-6, rtol=1e-4), (
		f"Failed on weight grad: bias={bias}, pad={pad}, stride={stride}, ks={kernel_size}, dil={dilation}, groups={groups}"
	)
	if bias:
		assert torch.allclose(conv1dw.bias.grad, grad_b_ref, atol=1e-6, rtol=1e-4), (
			f"Failed on bias grad: bias={bias}, pad={pad}, stride={stride}, ks={kernel_size}, dil={dilation}, groups={groups}"
		)


@pytest.mark.unit
@pytest.mark.parametrize(
	"bias,padding,stride,kernel_size,groups",
	list(
		itertools.product(
			[True, False],  # bias
			[0, 1],  # padding
			[1, 2],  # stride
			[3, 5],  # kernel size
			[1, 2],  # groups
		)
	),
)
def test_conv_dw_match_conv_1d_complex(bias, padding, stride, kernel_size, groups):
	"""Test Conv1dDW with complex tensors (requires conjugate for gradient computation)"""
	B, C_in, L = 4, 8, 32
	C_out = 12

	# Skip if C_in or C_out not divisible by groups
	if C_in % groups != 0 or C_out % groups != 0:
		pytest.skip(f"C_in={C_in} or C_out={C_out} not divisible by groups={groups}")

	# Create complex input
	x = torch.randn((B, C_in, L), dtype=torch.complex64, requires_grad=True)

	# Reference conv with complex weights
	ref = nn.Conv1d(
		C_in,
		C_out,
		kernel_size,
		bias=bias,
		padding=padding,
		stride=stride,
		groups=groups,
		dtype=torch.complex64,
	)
	ref_out = ref(x)
	loss_ref = ref_out.abs().sum()  # Use abs().sum() as loss for complex tensors
	loss_ref.backward()
	grad_w_ref = ref.weight.grad.clone()
	grad_b_ref = ref.bias.grad.clone() if bias else None

	# Conv1dDW wrapper
	conv1d = nn.Conv1d(
		C_in,
		C_out,
		kernel_size,
		bias=bias,
		padding=padding,
		stride=stride,
		groups=groups,
		dtype=torch.complex64,
	)
	conv1d.load_state_dict(ref.state_dict())
	conv1dw = Conv1dDW(conv1d, device=torch.device("cpu"))

	x2 = x.clone().detach().requires_grad_(True)
	y = conv1dw(x2)
	assert torch.allclose(y, ref_out, atol=1e-5, rtol=1e-4), (
		f"Failed on fwd: bias={bias}, pad={padding}, stride={stride}, ks={kernel_size}, groups={groups}"
	)

	conv1dw.move_last_computed("input", 0)
	loss = y.abs().sum()
	loss.backward()

	assert torch.allclose(x2.grad, x.grad, atol=1e-5, rtol=1e-4), (
		f"Complex input gradient mismatch: max diff = {(x2.grad - x.grad).abs().max()}\n"
		f"Conv1dDX.backward must use weight.conj() for complex tensors"
	)
	conv1dw.move_last_computed("grad_output", 0)

	conv1dw.backward(0)

	assert torch.allclose(conv1dw.weight.grad, grad_w_ref, atol=1e-5, rtol=1e-4), (
		f"Complex weight gradient mismatch: max diff = {(conv1dw.weight.grad - grad_w_ref).abs().max()}, "
		f"bias={bias}, pad={padding}, stride={stride}, ks={kernel_size}, groups={groups}"
	)
	if bias:
		assert torch.allclose(conv1dw.bias.grad, grad_b_ref, atol=1e-5, rtol=1e-4), (
			f"Complex bias gradient mismatch: max diff = {(conv1dw.bias.grad - grad_b_ref).abs().max()}, "
			f"bias={bias}, pad={padding}, stride={stride}, ks={kernel_size}, groups={groups}"
		)
