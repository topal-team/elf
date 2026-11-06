"""
FLOP formulas for the operations ``torch.utils.flop_counter`` does not know about.

Importing this module registers them, which is a global side effect on torch: any
``FlopCounterMode`` in the process starts counting GELU and FFT afterwards. ``elf/__init__.py``
imports it for that reason alone, so ``import elf`` is enough to get the counts right --
``examples/train_fno.py`` relies on it.
"""

import math

import torch

from torch.utils.flop_counter import register_flop_formula

from elf.utils import prod

__all__ = [
	"ewise_binary_flops",
	"ewise_unary_flops",
	"gelu_flops",
	"gelu_backward_flops",
	"fft_flops_c2c",
	"fft_flops_r2c",
]

aten = torch.ops.aten


@register_flop_formula([aten.add, aten.mul, aten.sub, aten.div])
def ewise_binary_flops(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
	return prod(a_shape)


@register_flop_formula([aten.abs])
def ewise_unary_flops(a_shape, *args, out_shape=None, **kwargs) -> int:
	return prod(a_shape)


@register_flop_formula(aten.gelu)
def gelu_flops(x_shape, *args, **kwargs) -> int:
	# GELU with approximate='none' (default) uses: 0.5 * x * (1 + erf(x / sqrt(2)))
	# Breaking down operations per element:
	# - x / sqrt(2): div (1 FLOP)
	# - erf(...): error function (~30 FLOPs for approximation)
	# - 1 + erf(...): add (1 FLOP)
	# - x * (...): mul (1 FLOP)
	# - 0.5 * (...): mul (1 FLOP)
	# Total: ~34 FLOPs per element
	return 34 * prod(x_shape)


@register_flop_formula([aten.gelu_backward])
def gelu_backward_flops(x_shape, *args, **kwargs) -> int:
	return gelu_flops(x_shape)  # it's not significantly different from the forward pass


@register_flop_formula([aten._fft_c2c])
def fft_flops_c2c(x_shape, dim, *args, **kwargs) -> int:
	# Complex-to-complex FFT: 5 * N * log2(N) FLOPs per dimension
	# where N is the size of the dimension being transformed
	total_flops = 0
	dims = dim if isinstance(dim, (list, tuple)) else [dim]

	for d in dims:
		n = x_shape[d]
		if n > 1:
			total_flops += 5 * n * math.log2(n)

	# Multiply by the number of elements in all other dimensions
	other_dims_size = prod(x_shape) // prod(x_shape[d] for d in dims)
	return int(total_flops * other_dims_size)


@register_flop_formula([aten._fft_r2c])
def fft_flops_r2c(x_shape, dim, *args, **kwargs) -> int:
	return fft_flops_c2c(x_shape, dim) // 2
