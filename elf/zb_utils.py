import torch
import torch.nn as nn


class LinearDX(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, linear):
		ctx.linear = linear
		linear.last_input = input
		ctx.save_for_backward(input) # used in dL/dw, we mark it as saved here to force torch.utils.checkpoint to recompute it
		return torch.nn.functional.linear(input, linear.weight, linear.bias)

	@staticmethod
	def backward(ctx, grad_output):
		linear = ctx.linear
		linear.last_grad_output = grad_output
		with torch.no_grad():
			grad_input = torch.matmul(grad_output, linear.weight)

		# del ctx.linear # don't delete in case of multiple backward calls
		return grad_input, None


class LayerDW(nn.Module):
	"""
	Abstract class to support decoupled backwards for inputs and parameters.
	The idea is to replace default nn.Modules by a custom autograd function that only computes gradients with respect to its inputs during backward pass, and saves the incoming gradients for later. Then, the call to LayerDW's .backward() uses these saved gradients to compute gradients for its parameters.
	For a concrete example, see LinearDW and LinearDX.
	"""

	def __init__(self):
		super(LayerDW, self).__init__()
		self.ctx = {"input": [], "grad_output": []}

	def forward(self, *args, **kwargs):
		raise NotImplementedError("Subclasses must implement forward()")

	def backward(self):
		"""
		Compute and assign gradients for layer parameters using saved context.
		Must be implemented by subclasses.
		"""
		raise NotImplementedError("Subclasses must implement backward()")

	def clear(self):
		self.ctx["input"].clear()
		self.ctx["grad_output"].clear()

	def is_empty(self, queue):
		return all(x is None for x in self.ctx[queue])

	def _state(self):
		ninputs = len([x for x in self.ctx["input"] if x is not None])
		ngrads = len([x for x in self.ctx["grad_output"] if x is not None])
		return f"({ninputs},{ngrads})"

	def set(self, queue, idx, value):
		values = self.ctx[queue]
		if len(values) <= idx:
			values.extend([None] * (idx - len(values) + 1))
		if values[idx] is not None:
			raise ValueError(f"{queue} at index {idx} already set")
		values[idx] = value

	def delete(self, queue, idx):
		values = self.ctx[queue]
		if idx >= len(values) or values[idx] is None:
			raise ValueError(f"{queue} at index {idx} not set")
		values[idx] = None

	def move_last_computed(self, queue, idx):
		if getattr(self, f"last_{queue}", None) is None:
			raise ValueError(f"Last {queue} not set")
		self.set(queue, idx, getattr(self, f"last_{queue}"))
		setattr(self, f"last_{queue}", None)

	def offload_last(self, idx, to="cpu"):
		grads = self.ctx["grad_output"][idx]
		inputs = self.ctx["input"][idx]
		with torch.no_grad():
			self.ctx["grad_output"][idx] = grads.to(to, non_blocking=True)
			self.ctx["input"][idx] = inputs.to(to, non_blocking=True)


class LinearDW(nn.Linear, LayerDW):
	def __init__(self, in_features, out_features, bias=True):
		super(LinearDW, self).__init__(in_features, out_features, bias)

		# Execution order:
		# LinearDW.forward -> LinearDX.forward -> LinearDX.backward -> LinearDW.backward

	def forward(self, x, *args, **kwargs):
		return LinearDX.apply(x, self, *args, **kwargs)

	def backward(self, mb_id):
		assert len(self.ctx["grad_output"]) >= mb_id, "No grad kept for backward"
		assert len(self.ctx["input"]) >= mb_id, "No input kept for backward"
		grad_output = self.ctx["grad_output"][mb_id]
		inputs = self.ctx["input"][mb_id]
		assert grad_output is not None, f"Grad output not set for mb {mb_id}"
		assert inputs is not None, f"Input not set for mb {mb_id}"

		with torch.no_grad():
			go = grad_output.reshape(-1, grad_output.size(-1))
			inp = inputs.reshape(-1, inputs.size(-1))

			# dL/dW
			grads_w = torch.matmul(go.T, inp).to(self.weight.device, non_blocking=True)
			self.weight.grad = (
				grads_w if self.weight.grad is None else self.weight.grad + grads_w
			)  # accumulate

			# dL/db
			if self.bias is not None:
				grads_b = go.sum(0).to(self.bias.device, non_blocking=True)
				self.bias.grad = (
					grads_b if self.bias.grad is None else self.bias.grad + grads_b
				)  # accumulate

		self.ctx["grad_output"][mb_id] = None
		self.ctx["input"][mb_id] = None


def replace_linear_with_linear_dw(model, device):
	"""
	Replace all nn.Linear modules in the model with LinearDW, inplace.
	"""
	for name, module in model.named_modules():
		if isinstance(module, nn.Linear) and not isinstance(module, LayerDW):
			if "." not in name:
				parent = model
			else:
				parent = model.get_submodule(name[: name.rfind(".")])
			child = name.split(".")[-1]
			new_module = LinearDW(module.in_features, module.out_features, module.bias is not None).to(
				device
			)
			new_module.weight.data = module.weight.data  # avoid copying data
			if module.bias is not None:
				new_module.bias.data = module.bias.data
			setattr(parent, child, new_module)


class LayerDWTracer(torch.fx.Tracer):
	"""
	Tracer that treats LayerDW modules as leaf modules to prevent tracing their internals.
	"""

	def is_leaf_module(self, m: torch.nn.Module, *args, **kwargs) -> bool:
		# Treat LayerDW modules as leaf modules to prevent tracing their internals
		if isinstance(m, LayerDW):
			return True
		# Use default behavior for other modules
		return super().is_leaf_module(m, *args, **kwargs)
