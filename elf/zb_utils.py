"""
Utility functions for decoupled backward passes, based on Zero-Bubble.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn


__all__ = [
	"LayerDW",
	"recording",
	"LinearDW",
	"Conv1dDW",
	"LinearDX",
	"Conv1dDX",
	"LayerDWTracer",
	"replace_layer_with_layer_dw",
	"replace_linear_with_linear_dw",
	"partial_dx_recomputation",
]


_RECORDING = (True, True)  # inputs, gradients


@contextmanager
def recording(gradients=True, inputs=True):
	"""
	Gate what a backward passing through :class:`LayerDW` layers keeps for ``W``.

	``W`` needs one input and one output gradient per use, and whatever another operation is going
	to rebuild is not worth keeping now.

	:param gradients: whether a backward run inside this context records anything at all
	:type gradients: bool
	:param inputs: whether it keeps the input beside each gradient it records
	:type inputs: bool
	"""
	global _RECORDING
	previous = _RECORDING
	_RECORDING = (inputs, gradients)
	try:
		yield
	finally:
		_RECORDING = previous


def _accumulate_grads(tensor, grads):
	if tensor.grad is None:
		tensor.grad = grads
	else:
		# we need to clone the gradient in case it has overlapping memory (https://discuss.pytorch.org/t/about-error-more-than-one-element-of-the-written-to-tensor-refers-to-a-single-memory-location/85526)
		if torch._debug_has_internal_overlap(tensor.grad):
			tensor.grad = tensor.grad.clone().detach()
		with torch.no_grad():
			tensor.grad.add_(grads)


class _Use:
	"""
	One application of a :class:`LayerDW`, shared by the two nodes it puts in the graph.

	The layer's own node sees the gradient that comes back and :class:`_KeepInput`, ahead of it,
	the input. A backward reaches the first one first, so it opens the pair it records and leaves
	it here for the second one to complete.
	"""

	__slots__ = ("pending",)

	def __init__(self):
		self.pending = None


class _KeepInput(torch.autograd.Function):
	"""
	Save the input of one use of a :class:`LayerDW`, in a node of its own ahead of the layer.

	A function saves its tensors only once its forward has returned, so a layer saving its own
	input would already have computed its output by then. Saved here, the input is in the graph
	before the layer runs, and a recomputation that only needed that input can stop before it.

	The weight is passed so that the node exists, and its backward runs, even when the input
	does not require grad. It is also what a partial backward differentiates against to reach it.
	"""

	@staticmethod
	def forward(ctx, input, weight, layer, use):
		ctx.use = use
		ctx.save_for_backward(input)
		layer.fill_input(input)  # no-op unless this forward is recomputing inputs W is waiting for
		return input.view_as(input)

	@staticmethod
	def backward(ctx, grad):
		pair, ctx.use.pending = ctx.use.pending, None
		if pair is not None:
			(input,) = ctx.saved_tensors
			pair[0] = input.detach()
		return grad, None, None, None


class LinearDX(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, bias, linear, use):
		# We need to pass both weight/bias AND linear to backward, because
		# linear is reached by a side-effect, and if we don't have input tensors that require grad,
		# autograd will not automatically mark the output as needing grad.
		ctx.linear = linear
		ctx.use = use
		return torch.nn.functional.linear(input, weight, bias)

	@staticmethod
	def backward(ctx, grad_output):
		linear = ctx.linear
		linear.record(ctx.use, grad_output)

		# Compute input gradient: ∂L/∂x = ∂L/∂y @ ∂y/∂x
		weight = linear.weight.conj() if grad_output.is_complex() else linear.weight

		if grad_output.dim() == 2:
			grad_input = torch.mm(grad_output, weight)
		else:
			grad_input = torch.matmul(grad_output, weight)

		# del ctx.linear # don't delete in case of multiple backward calls
		return grad_input, None, None, None, None


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
		self.pairs = []  # one entry per micro-batch
		self.last_pairs = None  # what the last backward recorded, not yet assigned to a micro-batch
		self.filling = None  # [micro-batch, next use] while a recomputed forward fills inputs back

	@classmethod
	def register_dx_function(cls, dx_class):
		"""
		Register a DX autograd function class for this LayerDW subclass.

		:param dx_class: The corresponding autograd.Function class (e.g., LinearDX)
		:type dx_class: type
		"""
		cls._dx_function_registry[cls] = dx_class
		return dx_class

	def backward(self, mb_id: int):
		"""
		Compute and assign gradients for layer parameters using saved context.
		Must be implemented by subclasses.

		:param mb_id: Micro-batch index to compute gradients for
		:type mb_id: int
		"""
		raise NotImplementedError("Subclasses must implement backward()")

	def clear(self):
		"""
		Delete all saved tensors.
		"""
		self.pairs.clear()
		self.last_pairs = None
		self.filling = None

	def is_empty(self, mb_id):
		"""
		Check whether anything was recorded for a micro-batch. A module the pass did not use
		records nothing, and has no parameter gradient to compute.

		:param mb_id: micro-batch index to check
		:type mb_id: int
		:return: True if nothing is kept for that micro-batch
		:rtype: bool
		"""
		return mb_id >= len(self.pairs) or self.pairs[mb_id] is None

	def _state(self):
		"""
		Return the number of uses recorded per micro-batch, as a string.
		"""
		return f"({[len(x) if x is not None else 0 for x in self.pairs]})"

	def record(self, use, grad_output):
		"""
		Keep one use of this layer: the input it was applied to, and the gradient that came back
		for it. A layer applied several times in a pass contributes one term per use to its
		parameter gradient, so the two halves have to stay together.

		The gradient is all the layer's own node has. The input is filled in by the node that
		saved it, which the backward reaches next.

		:param use: the use the gradient came back for
		:type use: _Use
		:param grad_output: gradient with respect to the corresponding output
		:type grad_output: torch.Tensor
		"""
		keep_input, keep_gradient = _RECORDING
		if not keep_gradient:
			return

		if self.last_pairs is None:
			self.last_pairs = []

		pair = [None, grad_output]
		if keep_input:
			use.pending = pair
		self.last_pairs.append(pair)

	def needs_inputs(self, mb_id):
		"""
		Whether a recomputed forward still has to give this micro-batch its inputs back.

		:param mb_id: micro-batch index to check
		:type mb_id: int
		:rtype: bool
		"""
		if self.is_empty(mb_id):
			return False

		return any(input is None for input, _ in self.pairs[mb_id])

	def begin_filling(self, mb_id):
		"""
		Start putting back the inputs of a micro-batch, one per use, in the order a forward
		produces them.

		:param mb_id: micro-batch index whose inputs are being recomputed
		:type mb_id: int
		"""
		self.filling = [mb_id, 0]

	def fill_input(self, input):
		"""
		Give one use its input back, the uses being filled in the order they are recomputed.

		:param input: input of the use being recomputed
		:type input: torch.Tensor
		"""
		if self.filling is None:
			return

		mb_id, index = self.filling
		pairs = self.pairs[mb_id]
		if index >= len(pairs):
			raise RuntimeError(
				f"the recomputed forward used this layer more than the {len(pairs)} time(s) the "
				"original one did, so its uses can no longer be told apart"
			)

		pairs[index][0] = input.detach()
		self.filling[1] = index + 1

	def end_filling(self):
		"""
		Stop putting inputs back, and check that every use got one.

		:raises RuntimeError: if the recomputed forward used this layer fewer times than the
			original one did
		"""
		mb_id, index = self.filling
		self.filling = None

		expected = 0 if self.is_empty(mb_id) else len(self.pairs[mb_id])
		if index != expected:
			raise RuntimeError(
				f"the recomputed forward used this layer {index} time(s) where the original one "
				f"used it {expected} time(s), so its uses can no longer be told apart"
			)

	def commit(self, mb_id):
		"""
		Assign what the last backward recorded to a micro-batch.

		A backward walks the uses of a layer in the reverse of the order the forward produced
		them, so what it recorded is reversed here: the uses are kept in forward order, which is
		the order a recomputed forward fills their inputs back in through :meth:`fill_input`.

		Recording nothing is not an error: a module the pass did not use is simply left alone.

		:param mb_id: micro-batch index the last backward belongs to
		:type mb_id: int
		:raises ValueError: if that micro-batch already holds something
		"""
		if self.last_pairs is None:
			return

		if len(self.pairs) <= mb_id:
			self.pairs.extend([None] * (mb_id - len(self.pairs) + 1))
		if self.pairs[mb_id] is not None:
			raise ValueError(f"pairs at index {mb_id} already set")

		self.last_pairs.reverse()
		self.pairs[mb_id] = self.last_pairs
		self.last_pairs = None


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
		use = _Use()
		x = _KeepInput.apply(x, self.weight, self, use)
		return LinearDX.apply(x, self.weight, self.bias, self, use, *args, **kwargs)

	def backward(self, mb_id):
		pairs = self.pairs[mb_id]
		assert pairs is not None, f"Nothing recorded for mb {mb_id}"
		self.pairs[mb_id] = None

		for inputs, grad_output in pairs:
			assert inputs is not None, (
				f"Input dropped for mb {mb_id} and never recomputed, recompute the forward first"
			)

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
			_accumulate_grads(self.weight, grads_w)

			# dL/db
			if self.bias is not None:
				grads_b = go.sum(0)
				_accumulate_grads(self.bias, grads_b)


LinearDW.register_dx_function(LinearDX)


class Conv1dDX(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, weight, bias, conv1d, use):
		ctx.conv1d = conv1d
		ctx.use = use
		ctx.input_shape = input.shape

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
		input_shape = ctx.input_shape
		conv1d.record(ctx.use, grad_output)

		with torch.no_grad():
			# Compute grad_input using transposed convolution
			# This is the standard way to compute input gradients for conv
			stride = conv1d.stride[0] if isinstance(conv1d.stride, tuple) else conv1d.stride
			padding = conv1d.padding[0] if isinstance(conv1d.padding, tuple) else conv1d.padding
			dilation = conv1d.dilation[0] if isinstance(conv1d.dilation, tuple) else conv1d.dilation
			kernel_size = conv1d.weight.shape[2]

			# Handle padding="same" by computing the actual padding used
			if isinstance(padding, str) and padding == "same":
				l_in = input_shape[2]
				l_out = grad_output.shape[2]
				padding_total = max(0, (l_out - 1) * stride + dilation * (kernel_size - 1) + 1 - l_in)
				padding = padding_total // 2

			# Compute output_padding to match input shape
			output_padding = input_shape[2] - (
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

		return grad_input, None, None, None, None


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
		use = _Use()
		x = _KeepInput.apply(x, self.weight, self, use)
		return Conv1dDX.apply(x, self.weight, self.bias, self, use, *args, **kwargs)

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
		pairs = self.pairs[mb_id]
		assert pairs is not None, f"Nothing recorded for mb {mb_id}"
		self.pairs[mb_id] = None

		for inputs, grad_output in pairs:
			assert inputs is not None, (
				f"Input dropped for mb {mb_id} and never recomputed, recompute the forward first"
			)

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

			_accumulate_grads(self.weight, grads_w)

			if self.bias is not None:
				grads_b = grad_output.sum(dim=(0, 2))

				if grads_b.device != self.bias.device:
					grads_b = grads_b.to(self.bias.device, non_blocking=True)

				_accumulate_grads(self.bias, grads_b)


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
	Check if the given grad_fn is a node a LayerDW put in the graph: its DX function (LinearDX,
	Conv1dDX, etc.), found through the LayerDW registry, or the node keeping its input.
	"""
	grad_fn_type = type(grad_fn)

	for dx_class in [_KeepInput, *LayerDW._dx_function_registry.values()]:
		if grad_fn_type.__name__ == f"{dx_class.__name__}Backward":
			return True

	return False


def _get_gradient_edges_needed_for_w(grad_fn):
	"""
	Get the set of gradient edges (corresponding to tensors) that we need to differentiate against in order to fill all the pairs of LayerDW modules, gradients and inputs.

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
	outputs.backward(grad_outputs, inputs=gradient_edges, retain_graph=retain_graph)
