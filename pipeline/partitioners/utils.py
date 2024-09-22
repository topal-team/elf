import torch
import torch.nn as nn
import torch.fx as fx

__all__ = ["RemoveInplaceTransformer"]

inplace_modules = (
	nn.ReLU,
	nn.LeakyReLU,
	nn.ELU,
	nn.GELU,
	nn.CELU,
	nn.SELU,
	nn.PReLU,
	nn.RReLU,
	nn.Hardswish,
	nn.Hardshrink,
	nn.Hardsigmoid,
	nn.Hardtanh,
	nn.Softsign,
	nn.Softplus,
	nn.Mish,
	nn.SiLU,
	nn.Tanhshrink,
	nn.Threshold,
)

inplace_functions = (
	torch.abs_,
	torch.neg_,
	torch.exp_,
	torch.log_,
	torch.log10_,
	torch.log2_,
	torch.sqrt_,
	torch.rsqrt_,
	torch.ceil_,
	torch.floor_,
	torch.round_,
	torch.trunc_,
	torch.frac_,
	torch.sin_,
	torch.cos_,
	torch.tan_,
	torch.asin_,
	torch.acos_,
	torch.atan_,
	torch.sinh_,
	torch.cosh_,
	torch.tanh_,
	torch.asinh_,
	torch.acosh_,
	torch.atanh_,
	torch.sigmoid_,
	torch.zero_,
	torch.clamp_,
	torch.fill_,
	torch.erf_,
	torch.erfc_,
	torch.expm1_,
	torch.log1p_,
	torch.reciprocal_,
	torch.rsqrt_,
)


class RemoveInplaceTransformer(fx.Transformer):
	"""
	A transformer that removes in-place operations from a PyTorch model.
	"""

	def __init__(self, module):
		super().__init__(module)

	def call_function(self, target, args, kwargs):
		if target.__name__ in inplace_functions:
			out_of_place_func = getattr(torch, target.__name__[:-1])
			return out_of_place_func(*args, **kwargs)
		return super().call_function(target, args, kwargs)

	def call_module(self, target, args, kwargs):
		module = self.fetch_attr(target)
		if isinstance(module, (inplace_modules)):
			module.inplace = False
		return super().call_module(target, args, kwargs)
