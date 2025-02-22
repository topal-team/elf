import torch
import torch.nn as nn

class LinearDX(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, linear):
		ctx.linear = linear
		ctx.input = input  # if under torch.no_grad(), does this cause memory leaks?
		linear.ctx["input"].append(input.detach())
		return torch.nn.functional.linear(input, linear.weight, linear.bias)

	@staticmethod
	def backward(ctx, grad_output):
		linear = ctx.linear
		linear.ctx["grad_output"].append(grad_output.detach())
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
		self.ctx = {}

	def backward(self):
		"""
		Compute and assign gradients for layer parameters using saved context.
		Must be implemented by subclasses.
		"""
		raise NotImplementedError("Subclasses must implement backward()")


class LinearDW(nn.Linear, LayerDW):
	def __init__(self, in_features, out_features, bias=True):
		super(LinearDW, self).__init__(in_features, out_features, bias)
		self.ctx["input"] = []
		self.ctx["grad_output"] = []

		# Execution order:
		# LinearDW.forward -> LinearDX.forward -> LinearDX.backward -> LinearDW.backward

	def clear(self):
		self.ctx["input"].clear()
		self.ctx["grad_output"].clear()

	def forward(self, x, *args, **kwargs):
		return LinearDX.apply(x, self, *args, **kwargs)

	def offload_last(self, to="cpu"):
		grads = self.ctx["grad_output"].pop(-1)
		inputs = self.ctx["input"].pop(-1)
		with torch.no_grad():
			self.ctx["grad_output"].append(grads.to(to, non_blocking=True))
			self.ctx["input"].append(inputs.to(to, non_blocking=True))

	def del_last_act(self):
		"""
		Delete the last activation from the queue.
		"""
		self.ctx["input"].pop(-1)

	def del_last_grad(self):
		"""
		Delete the last gradient from the queue.
		"""
		self.ctx["grad_output"].pop(-1)

	def del_first_grad(self):
		"""
		Delete the first gradient from the queue.
		"""
		self.ctx["grad_output"].pop(0)

	def del_act_last_grad(self):
		"""
		Delete the last activation from the queue.
		"""
		idx = len(self.ctx["grad_output"]) - 1
		self.ctx["input"].pop(idx)

	def mark_last_act_as_recomputed(self):
		"""
		Move the last activation to the front
		"""
		last_act = self.ctx["input"].pop(-1)
		self.ctx["input"].insert(0, last_act)

	def mark_last_grad_as_recomputed(self):
		"""
		Move the last gradient to the front
		"""
		last_grad = self.ctx["grad_output"].pop(-1)
		self.ctx["grad_output"].insert(0, last_grad)

	def sync_input_grads(self):
		"""
		When using torch.utils.checkpoint, the inputs are recomputed just before the backward pass. Some other forwards may have been done inbetween. Therefore we need to sync their position in the queues.
		Intended to be called just after the backward() call on a checkpointed module.
		"""
		idx = len(self.ctx["grad_output"]) - 1
		x = self.ctx["input"].pop(-1)
		self.ctx["input"].insert(idx, x)

	def backward(self):
		assert len(self.ctx["grad_output"]) > 0, "No grad kept for backward"
		assert len(self.ctx["input"]) > 0, "No input kept for backward"
		grad_output = self.ctx["grad_output"].pop(0)
		inputs = self.ctx["input"].pop(0)

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
