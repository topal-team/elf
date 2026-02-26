"""
Utility functions for decoupled backward passes, based on Zero-Bubble.
"""

import torch
import torch.nn as nn


class LinearDX(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, bias, linear):
		# We need to pass both weight/bias AND linear to backward, because
		# We store the input in linear.last_input, with a side-effect,
		# but if we don't have input tensors that require grad, autograd will not automatically mark the output as needing grad.
		ctx.linear = linear
		linear._store_input(input)
		ctx.save_for_backward(
			input.detach()
		)  # used in dL/dw, we mark it as saved here to force torch.utils.checkpoint to recompute it
		return torch.nn.functional.linear(input, weight, bias)

	@staticmethod
	def backward(ctx, grad_output):
		linear = ctx.linear
		linear._accumulate_grad_output(grad_output)

		# Compute input gradient: ∂L/∂x = ∂L/∂y @ ∂y/∂x
		# For y = x @ W^T (PyTorch convention):
		#   - Real case: ∂L/∂x = ∂L/∂y @ W
		#   - Complex case: ∂L/∂x = ∂L/∂y @ W̄ (Wirtinger calculus)
		# See: https://pytorch.org/docs/stable/notes/autograd.html#complex-autograd-doc
		weight = linear.weight.conj() if grad_output.is_complex() else linear.weight

		if grad_output.dim() == 2:
			grad_input = torch.mm(grad_output, weight)
		else:
			grad_input = torch.matmul(grad_output, weight)

		# del ctx.linear # don't delete in case of multiple backward calls
		return grad_input, None, None, None


class LayerDW:
	"""
	Mixin class to support decoupled backwards for inputs and parameters.
	The idea is to replace default nn.Modules by a custom autograd function that only computes gradients with respect to its inputs during backward pass, and saves the incoming gradients for later. Then, the call to LayerDW's .backward() uses these saved gradients to compute gradients for its parameters.
	For a concrete example, see LinearDW and LinearDX.
	"""

	_dx_function_registry = {}

	def __init__(self, *args, **kwargs):
		# Cooperative multiple inheritance: call next in MRO
		# Always use meta device to avoid allocating memory for tensors that will be replaced
		with torch.device("meta"):
			super().__init__(*args, **kwargs)

		# Initialize mixin-specific attributes
		self.ctx = {"input": [], "grad_output": []}

	@classmethod
	def register_dx_function(cls, dx_class):
		"""
		Register a DX autograd function class for this LayerDW subclass.

		:param dx_class: The corresponding autograd.Function class (e.g., LinearDX)
		:type dx_class: type
		"""
		cls._dx_function_registry[cls] = dx_class
		return dx_class

	def backward(self):
		"""
		Compute and assign gradients for layer parameters using saved context.
		Must be implemented by subclasses.
		"""
		raise NotImplementedError("Subclasses must implement backward()")

	def clear(self):
		"""
		Delete all saved tensors.
		"""
		self.ctx["input"].clear()
		self.ctx["grad_output"].clear()
		self.last_input = None
		self.last_grad_output = None

	def is_empty(self, queue):
		"""
		Check if there are any saved tensors in the queue.

		:param queue: queue to check among ["input", "grad_output"]
		:type queue: str
		:return: True if there are no saved tensors in the queue, False otherwise
		:rtype: bool
		"""
		return all(x is None for x in self.ctx[queue])

	def _state(self):
		"""
		Return the state of the queues as a string.
		"""
		ninputs = len([x for x in self.ctx["input"] if x is not None])
		ngrads = len([x for x in self.ctx["grad_output"] if x is not None])
		return f"({ninputs},{ngrads})"

	def _store_input(self, input):
		"""
		Store input for later use in backward_params.
		Marks input as model input if it has no grad_fn (for checkpointing logic).

		:param input: input tensor to store
		:type input: torch.Tensor
		"""
		self.last_input = input.detach()
		# Mark if this is the input of the model (no grad_fn)
		# This prevents remat logic from deleting it since it won't be recomputed by torch.utils.checkpoint
		setattr(self.last_input, "is_model_input", input.grad_fn is None)

	def _accumulate_grad_output(self, grad_output):
		"""
		Accumulate grad_output for later use in backward_params.
		If module is reused multiple times in forward pass, this will be called multiple times
		and we need to sum all the gradients, not overwrite them.

		:param grad_output: gradient with respect to output
		:type grad_output: torch.Tensor
		"""
		if getattr(self, "last_grad_output", None) is None:
			self.last_grad_output = grad_output
		else:
			self.last_grad_output = self.last_grad_output + grad_output

	def set(self, queue, idx, value):
		"""
		Set a value in the queue at a given index.

		:param queue: queue to set the value in
		:type queue: str
		:param idx: index to set the value at
		:type idx: int
		:param value: value to set
		:type value: torch.Tensor
		"""
		values = self.ctx[queue]
		if len(values) <= idx:
			values.extend([None] * (idx - len(values) + 1))
		if values[idx] is not None:
			raise ValueError(f"{queue} at index {idx} already set")
		values[idx] = value

	def delete(self, queue, idx):
		"""
		Delete a value in the queue at a given index.

		:param queue: queue to delete the value from
		:type queue: str
		:param idx: index to delete the value at
		:type idx: int
		"""
		values = self.ctx[queue]
		if idx >= len(values) or values[idx] is None:
			raise ValueError(f"{queue} at index {idx} not set")
		values[idx] = None

	def move_last_computed(self, queue, idx):
		"""
		Move the last computed value to the queue at a given index.
		This should be called after the forward pass for every microbatch.

		:param queue: queue to move the last computed value to
		:type queue: str
		:param idx: index to move the last computed value to
		:type idx: int
		"""
		last_value = getattr(self, f"last_{queue}", None)
		if last_value is None:
			# raise ValueError(f"Last {queue} not set")
			return  # this can happen if a mpdule does not use some of its submodules
		self.set(queue, idx, last_value)
		setattr(self, f"last_{queue}", None)

	def offload_last(self, idx, to="cpu"):
		"""
		Offload the last computed value to a given device.
		"""
		grads = self.ctx["grad_output"][idx]
		inputs = self.ctx["input"][idx]
		with torch.no_grad():
			self.ctx["grad_output"][idx] = grads.to(to, non_blocking=True)
			self.ctx["input"][idx] = inputs.to(to, non_blocking=True)


# Note: inherit from LayerDW FIRST to make use of the mixin's initialization logic
class LinearDW(LayerDW, nn.Linear):
	def __init__(self, linear, *args, **kwargs):
		super().__init__(
			linear.in_features, linear.out_features, linear.bias is not None, *args, **kwargs
		)

		# Use weights and bias from the original layer
		self.weight = linear.weight
		if linear.bias is not None:
			self.bias = linear.bias

		# Execution order:
		# LinearDW.forward -> LinearDX.forward -> LinearDX.backward -> LinearDW.backward

	def forward(self, x, *args, **kwargs):
		return LinearDX.apply(x, self.weight, self.bias, self, *args, **kwargs)

	def backward(self, mb_id):
		grad_output = self.ctx["grad_output"][mb_id]
		inputs = self.ctx["input"][mb_id]

		# Flatten to 2D if needed (avoid reshape if already 2D)
		if grad_output.dim() > 2:
			go = grad_output.reshape(-1, grad_output.size(-1))
		else:
			go = grad_output

		if inputs.dim() > 2:
			inp = inputs.reshape(-1, inputs.size(-1))
		else:
			inp = inputs

		# Weight gradient: ∂L/∂W = ∂L/∂y^T @ x
		# For y = x @ W^T (PyTorch convention):
		#   - Real case: ∂L/∂W = (∂L/∂y)^T @ x
		#   - Complex case: ∂L/∂W = (∂L/∂y)^T @ x̄ (Wirtinger calculus)
		# This matches PyTorch's autograd for complex parameters
		if go.is_complex():
			grads_w = torch.mm(go.t(), inp.conj())
		else:
			grads_w = torch.mm(go.t(), inp)

		# Accumulate gradients
		if self.weight.grad is None:
			self.weight.grad = grads_w
		else:
			self.weight.grad.add_(grads_w)

		# dL/db
		if self.bias is not None:
			grads_b = go.sum(0)
			if self.bias.grad is None:
				self.bias.grad = grads_b
			else:
				self.bias.grad.add_(grads_b)

		# Clear saved tensors
		self.ctx["grad_output"][mb_id] = None
		self.ctx["input"][mb_id] = None


LinearDW.register_dx_function(LinearDX)


class Conv1dDX(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, bias, conv1d):
		ctx.conv1d = conv1d
		conv1d._store_input(input)
		ctx.save_for_backward(input.detach())

		return torch.nn.functional.conv1d(
			input,
			weight,
			bias,
			stride=conv1d.stride,
			padding=conv1d.padding,
			dilation=conv1d.dilation,
			groups=conv1d.groups,
		)

	@staticmethod
	def backward(ctx, grad_output):
		conv1d = ctx.conv1d
		conv1d._accumulate_grad_output(grad_output)

		with torch.no_grad():
			(input,) = ctx.saved_tensors

			# Compute grad_input using transposed convolution
			# This is the standard way to compute input gradients for conv
			stride = conv1d.stride[0] if isinstance(conv1d.stride, tuple) else conv1d.stride
			padding = conv1d.padding[0] if isinstance(conv1d.padding, tuple) else conv1d.padding
			dilation = conv1d.dilation[0] if isinstance(conv1d.dilation, tuple) else conv1d.dilation
			kernel_size = conv1d.weight.shape[2]

			# Handle padding="same" by computing the actual padding used
			if isinstance(padding, str) and padding == "same":
				l_in = input.shape[2]
				l_out = grad_output.shape[2]
				padding_total = max(0, (l_out - 1) * stride + dilation * (kernel_size - 1) + 1 - l_in)
				padding = padding_total // 2

			# Compute output_padding to match input shape
			output_padding = input.shape[2] - (
				(grad_output.shape[2] - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
			)

			# Input gradient via transposed convolution
			# For complex tensors, Wirtinger calculus requires conjugated weights:
			#   ∂L/∂x = conv_transpose(∂L/∂y, W̄) where W̄ is conjugate of weight
			# This matches PyTorch's complex autograd behavior
			weight = conv1d.weight.conj() if grad_output.is_complex() else conv1d.weight

			grad_input = torch.nn.functional.conv_transpose1d(
				grad_output,
				weight,
				stride=stride,
				padding=padding,
				output_padding=output_padding,
				dilation=dilation,
				groups=conv1d.groups,
			)

		return grad_input, None, None, None


class Conv1dDW(LayerDW, nn.Conv1d):
	def __init__(self, conv1d, *args, **kwargs):
		super().__init__(
			conv1d.in_channels,
			conv1d.out_channels,
			conv1d.kernel_size,
			stride=conv1d.stride,
			padding=conv1d.padding,
			dilation=conv1d.dilation,
			groups=conv1d.groups,
			bias=conv1d.bias is not None,
			*args,
			**kwargs,
		)

		# Use weights and bias from the original layer
		self.weight = conv1d.weight
		if conv1d.bias is not None:
			self.bias = conv1d.bias

	def forward(self, x, *args, **kwargs):
		return Conv1dDX.apply(x, self.weight, self.bias, self, *args, **kwargs)

	def _backward_weight_complex(self, inputs, grad_output, stride, padding, dilation, kernel_size):
		"""Compute weight gradient for complex tensors using Wirtinger calculus.

		The native convolution backward kernel doesn't support complex dtypes on CPU,
		so we use manual correlation with conjugated input: ∂L/∂W = (∂L/∂y)^T @ x̄.
		"""
		batch_size = inputs.shape[0]
		l_out = grad_output.shape[2]
		in_channels_per_group = self.in_channels // self.groups

		if padding > 0:
			input_padded = torch.nn.functional.pad(inputs, (padding, padding))
		else:
			input_padded = inputs

		stride_batch, stride_channel, stride_spatial = input_padded.stride()
		patches = input_padded.as_strided(
			size=(batch_size, self.in_channels, kernel_size, l_out),
			stride=(stride_batch, stride_channel, dilation * stride_spatial, stride * stride_spatial),
		)

		if self.groups == 1:
			grad_flat = grad_output.permute(0, 2, 1).reshape(-1, self.out_channels)
			patches_flat = patches.permute(0, 3, 1, 2).reshape(-1, self.in_channels * kernel_size)
			grads_w = torch.mm(grad_flat.t(), patches_flat.conj()).reshape(
				self.out_channels, self.in_channels, kernel_size
			)
		else:
			patches_grouped = patches.reshape(
				batch_size, self.groups, in_channels_per_group, kernel_size, l_out
			)
			grad_output_grouped = grad_output.reshape(
				batch_size, self.groups, self.out_channels // self.groups, l_out
			)
			grads_w = torch.einsum("ngol,ngikl->goik", grad_output_grouped, patches_grouped.conj())
			grads_w = grads_w.reshape(self.out_channels, in_channels_per_group, kernel_size)

		return grads_w

	def backward(self, mb_id):
		assert len(self.ctx["grad_output"]) > mb_id, "No grad kept for backward"
		assert len(self.ctx["input"]) > mb_id, "No input kept for backward"
		grad_output = self.ctx["grad_output"][mb_id]
		inputs = self.ctx["input"][mb_id]
		assert grad_output is not None, f"Grad output not set for mb {mb_id}"
		assert inputs is not None, f"Input not set for mb {mb_id}"

		with torch.no_grad():
			stride = self.stride[0] if isinstance(self.stride, tuple) else self.stride
			padding = self.padding[0] if isinstance(self.padding, tuple) else self.padding
			dilation = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
			kernel_size = self.weight.shape[2]

			# Handle padding="same" by computing the actual padding used
			if isinstance(padding, str) and padding == "same":
				l_in = inputs.shape[2]
				l_out = grad_output.shape[2]
				padding_total = max(0, (l_out - 1) * stride + dilation * (kernel_size - 1) + 1 - l_in)
				padding = padding_total // 2

			if inputs.is_complex():
				grads_w = self._backward_weight_complex(
					inputs, grad_output, stride, padding, dilation, kernel_size
				)
			else:
				grads_w = torch.nn.grad.conv1d_weight(
					inputs,
					self.weight.shape,
					grad_output,
					stride=[stride],
					padding=[padding],
					dilation=[dilation],
					groups=self.groups,
				)

		if grads_w.device != self.weight.device:
			grads_w = grads_w.to(self.weight.device, non_blocking=True)

		if self.weight.grad is None:
			self.weight.grad = grads_w
		else:
			self.weight.grad.add_(grads_w)  # accumulate without re-allocating

		if self.bias is not None:
			grads_b = grad_output.sum(dim=(0, 2))

			if grads_b.device != self.bias.device:
				grads_b = grads_b.to(self.bias.device, non_blocking=True)

			if self.bias.grad is None:
				self.bias.grad = grads_b
			else:
				self.bias.grad.add_(grads_b)  # accumulate without re-allocating

		self.ctx["grad_output"][mb_id] = None
		self.ctx["input"][mb_id] = None


Conv1dDW.register_dx_function(Conv1dDX)


def replace_linear_with_linear_dw(model, device):
	"""
	    Replace all nn.Linear modules in the model with LinearDW for Zero Bubble schedules.

	    :param model: model containing nn.Linear layers
	    :type model: nn.Module
	    :param device: Device for the LinearDW modules ("cuda", "cpu", or "meta")
	    :type device: str

	    Example:
	            >>> with torch.device("meta"):
	            ...     model = TransformerModel()
	            ...     replace_linear_with_linear_dw(model, "meta")
	            >>> pipe = Pipeline(model, sample, scheduler="zbh2")

	    .. note::
	            - Only needed for ZB schedulers: zbh1, zbh2, zbv
	            - Operation is performed in-place

	    .. seealso::
	            LinearDW: The replacement linear layer implementation

	.. deprecated:: current version
	    Use replace_layer_with_layer_dw instead.

	"""
	for name, module in model.named_modules():
		if isinstance(module, nn.Linear) and not isinstance(module, LayerDW):
			if "." not in name:
				parent = model
			else:
				parent = model.get_submodule(name[: name.rfind(".")])
			child = name.split(".")[-1]
			new_module = LinearDW(module, device=device)

			setattr(parent, child, new_module)


MappingDW = {nn.Linear: LinearDW, nn.Conv1d: Conv1dDW}


def replace_layer_with_layer_dw(model, only=None):
	"""
	Replace all nn.Modules in the model with their LayerDW equivalent if it exists, for Zero Bubble schedules.

	:param model: model containing nn.Modules
	:type model: nn.Module
	:param only: Only replace modules of these types
	:type only: list[type]
	"""
	for name, module in model.named_modules():
		if (
			isinstance(module, nn.Module)
			and type(module) in MappingDW
			and not isinstance(module, LayerDW)
			and (only is None or type(module) in only)
		):
			if "." not in name:
				parent = model
			else:
				parent = model.get_submodule(name[: name.rfind(".")])
			child = name.split(".")[-1]
			new_module = MappingDW[type(module)](module)

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


def _grad_fn_is_layer_dx(grad_fn):
	"""
	Check if the given grad_fn is a LayerDX-related node (LinearDX, Conv1dDX, etc.).
	Uses the LayerDW registry to detect registered DX autograd functions.
	"""
	grad_fn_type = type(grad_fn)

	for dx_class in LayerDW._dx_function_registry.values():
		if grad_fn_type.__name__ == f"{dx_class.__name__}Backward":
			return True

	return False


def _get_gradient_edges_needed_for_w(grad_fn):
	"""
	Get the set of gradient edges (corresponding to tensors) that we need to differentiate against in order to fill all "grad_output" fields of LayerDW modules.

	:param grad_fn: The grad_fn to start the search from. (Usually the loss.grad_fn)
	:type grad_fn: torch.autograd.Function
	:return: The set of gradient edges (corresponding to tensors) that we need to differentiate against in order to fill all "grad_output" fields of LayerDW modules
	:rtype: list[torch.autograd.graph.GradientEdge]
	"""
	seen = set()
	accumulate_nodes = []

	stack = [(grad_fn, False)]

	# Interesting fact:
	# this function was previously written recursively, but doing that creates a reference cycles (from closure to itself)
	# this should be managed by python's generational garbage collector, but it contains references to cpp objects (the grad_fn)
	# which cannot be properly freed. That ultimately lead to a (GPU!) memory leak.
	while stack:
		node, has_linear_dx_predecessor = stack.pop()
		if node in seen:
			continue

		seen.add(node)
		current_is_linear_dx = _grad_fn_is_layer_dx(node)
		if len(node.next_functions) == 0:
			if has_linear_dx_predecessor:
				accumulate_nodes.append(node)
			continue

		for fn in node.next_functions:
			if fn[0] is not None and fn[0] not in seen:
				stack.append((fn[0], current_is_linear_dx))

	gradient_edges = [
		torch.autograd.graph.get_gradient_edge(node.variable) for node in accumulate_nodes
	]

	return gradient_edges


def partial_dx_recomputation(outputs, grad_outputs=None, retain_graph=False):
	"""
	Recompute the minimal backward pass needed to compute gradients w.r.t. the parameters afterwards.
	"""
	if grad_outputs is None:
		assert outputs.numel() == 1, "Implicit gradient outputs are only supported for scalar outputs"
		grad_outputs = torch.ones_like(outputs)

	gradient_edges = _get_gradient_edges_needed_for_w(outputs.grad_fn)

	if len(gradient_edges) == 0:
		return  # nothing to do

	# allow_unused because grads wrt those nodes will be None, which PyTorch does not like
	torch.autograd.grad(
		outputs,
		inputs=gradient_edges,
		grad_outputs=grad_outputs,
		allow_unused=True,
		retain_graph=retain_graph,
	)
