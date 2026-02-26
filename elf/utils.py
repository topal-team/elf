"""
Various useful classes / functions
"""

import math
import time
import logging

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.flop_counter import register_flop_formula

from typing import List

logger = logging.getLogger(__name__)

dtypes = [
	torch.float16,
	torch.bfloat16,
	torch.float32,
	torch.float64,
	torch.int16,
	torch.int32,
	torch.int64,
	torch.complex32,
	torch.complex64,
	torch.complex128,
	torch.bool,
]


def _is_mpi():
	return dist.is_initialized() and dist.get_backend() == dist.Backend.MPI


def pretty_print_params(n):
	"""
	Format a number of parameters into a human-readable string.
	"""
	if n > 1e9:
		return f"{n / 1e9:.1f}B"
	elif n > 1e6:
		return f"{n / 1e6:.1f}M"
	else:
		return f"{int(n)}"


class Placement(List[int]):
	"""
	Device placement pattern for pipeline stages.

	A placement is a list where each element specifies which device (rank) hosts
	that pipeline stage. Different schedulers require different placement patterns:

	- Linear: [0, 1, 2, 3] - each stage on a different device
	- Interleaved: [0, 1, 2, 3, 0, 1, 2, 3] - multiple stages per device
	- V-schedule: [0, 1, 2, 3, 3, 2, 1, 0] - bidirectional pattern

	Example:
		>>> # Manual placement
		>>> placement = Placement([0, 1, 2, 3])
		>>>
		>>> # Automatic placement for scheduler
		>>> placement = Placement.default("zbh2", pp=4)  # [0, 1, 2, 3]
		>>> placement = Placement.default("zbv", pp=4)    # [0, 1, 2, 3, 3, 2, 1, 0]
	"""

	def __init__(self, placement: List[int]):
		super().__init__(placement)
		assert not dist.is_initialized() or max(self) < dist.get_world_size(), (
			"Placement is out of bounds"
		)

	def get_ids(self, rank: int) -> List[int]:
		"""
		Get the stage IDs assigned to a specific rank.

		Args:
			rank: Device rank to query

		Returns:
			List of stage indices on the specified rank

		Example:
			>>> placement = Placement([0, 1, 1, 2])
			>>> placement.get_ids(1)  # Stages on rank 1
			[1, 2]
		"""
		return [i for i in range(len(self)) if self[i] == rank]

	@staticmethod
	def default(scheduler, pp):
		"""
		Get the default placement pattern for a scheduler.

		Different pipeline schedulers have different placement requirements:

		- 1f1b, gpipe, zbh1, zbh2: Linear placement [0, 1, 2, ...]
		- megatron: Interleaved placement [0, 1, 2, ..., 0, 1, 2, ...]
		- hanayo, zbv: V-schedule [0, 1, 2, ..., 2, 1, 0]

		Args:
			scheduler: Scheduler name (str) or scheduler function
			pp: Number of pipeline parallel devices

		Returns:
			Placement object with the appropriate pattern

		Example:
			>>> # For standard schedulers
			>>> placement = Placement.default("1f1b", pp=4)  # [0, 1, 2, 3]
			>>>
			>>> # For interleaved schedules
			>>> placement = Placement.default("megatron", pp=4)  # [0, 1, 2, 3, 0, 1, 2, 3]
			>>>
			>>> # For V-schedules
			>>> placement = Placement.default("zbv", pp=4)  # [0, 1, 2, 3, 3, 2, 1, 0]
		"""
		if callable(scheduler):
			logger.warning(
				"Placement.default() expects a scheduler name, not the scheduler object itself. Using the default placement."
			)
			scheduler = ""

		if scheduler == "hanayo" or scheduler == "zbv":
			return Placement([i for i in range(pp)] + list(reversed([i for i in range(pp)])))
		elif scheduler == "megatron":
			return Placement([i for i in range(pp)] * 2)
		else:
			return Placement([i for i in range(pp)])

	def head(self):
		return self[0]

	def tail(self):
		return self[-1]

	def add_offset(self, offset: int):
		return Placement([i + offset for i in self])

	def is_tail(self, rank: int):
		return rank == self[-1]

	def is_head(self, rank: int):
		return rank == self[0]


class TensorMetadata:
	"""
	Informations about Tensors that are sent and received in p2p communication
	[dtype, *shape]
	"""

	MAX_SIZE = 16

	@staticmethod
	def from_tensor(t):
		"""
		Creates a TensorMetadata object from its Tensor equivalent (should be used when receiving metadata via p2p)

		:param t: Tensor representation of a metadata
		:type t: Tensor

		:return: corresponding metadata
		:rtype: TensorMetadata
		"""
		dtype = dtypes[int(t[0].item())]
		shape = []
		assert len(t.shape) == 1, "Metadata should only have one dimension"
		for s in t[1:]:
			s = int(s.item())
			if s == 0:
				break
			shape.append(s)

		metadata = TensorMetadata(torch.empty(0, dtype=dtype))
		metadata.shape = shape
		return metadata

	def __init__(self, t):
		"""
		Extract metadata from a tensor

		:param t: tensor
		:type t: Tensor
		"""
		self.shape = list(t.shape)
		self.dtype = t.dtype
		self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

	def to_tensor(self):
		"""
		Creates the Tensor representation of this metadata. Should be used when sending metadata via p2p

		:return: tensor representation of this metadata
		:rtype: Tensor
		"""
		t = torch.zeros(TensorMetadata.MAX_SIZE, device=self.device)
		t[0] = dtypes.index(self.dtype)
		for i, s in enumerate(self.shape):
			t[1 + i] = s
		return t

	def get_buffer(self, batch_size):
		"""
		Allocates a tensor with the right shape and dtype for this metadata

		:return: tensor corresponding to this metadata
		:rtype: Tensor
		"""
		buffer = torch.empty((batch_size, *self.shape), dtype=self.dtype, device=self.device)
		return buffer

	def __repr__(self):
		return f"TensorMetadata({self.shape})"

	def __str__(self):
		return f"TensorMetadata({self.shape})"


class NameMapping:
	"""
	Mapping between input/output variable names of two blocks
	"""

	def __init__(self, inputs, outputs):
		self.inputs_to_outputs = {inp: out for inp, out in zip(inputs, outputs)}
		self.outputs_to_inputs = {out: inp for inp, out in zip(inputs, outputs)}

	def to_output(self, name):
		return self.inputs_to_outputs[name]

	def to_input(self, name):
		return self.outputs_to_inputs[name]

	def __repr__(self):
		s = "NameMapping: (\n"
		for inp, out in self.inputs_to_outputs.items():
			s += f"  {inp} <-> {out}\n"
		s += ")"
		return s

	def __str__(self):
		return "NameMapping"


class Timer:
	"""
	Utility context to time the execution of some code, in seconds.
	Uses GPU if cuda is available, else times CPU execution

	Utilisation example: ::

		with Timer() as timer:
			# do some stuff
			# everything in this context will be timed
		# if you do something here, it will not be timed
		print(timer.time())

	"""

	def __new__(cls, type=None, name=None, *, type_=None, name_=None):
		timer_type = type or type_
		timer_name = name or name_

		if timer_type:
			if timer_type.lower() in ["gpu", "cuda"]:
				return TimerGPU(timer_name)
			elif timer_type.lower() == "cpu":
				return TimerCPU(timer_name)

		if torch.cuda.is_available():
			return TimerGPU(timer_name)
		else:
			return TimerCPU(timer_name)


class TimerCPU:
	"""
	Timer for CPU execution
	"""

	def __init__(self, name="unknown"):
		self.name = name

	def __enter__(self):
		self.start = time.perf_counter()
		return self

	def __exit__(self, *_):
		self.end = time.perf_counter()

	def time(self):
		"""
		Get the time elapsed in seconds.
		"""
		return self.end - self.start


class TimerGPU:
	"""
	Timer for CUDA execution
	"""

	def __init__(self, name="unknown"):
		self.name = name
		self.start_event = torch.cuda.Event(enable_timing=True)
		self.end_event = torch.cuda.Event(enable_timing=True)

	def __enter__(self):
		self.start_event.record()
		return self

	def __exit__(self, *_):
		self.end_event.record()

	def time(self):
		"""
		Get the time elapsed in seconds.
		"""
		self.end_event.synchronize()
		return self.start_event.elapsed_time(self.end_event) / 1000


def send_models(models, dst, group=None):
	"""
	Send a list of models using p2p comms
	Send the model structure (nn.Module object, parameters metadata, etc) via pickled CPU comm, and the actual tensors via GPU comm.

	:param models: list of models to send
	:type models: list[nn.Module]
	:param dst: destination rank
	:type dst: int
	:param group: communication group
	:type group: dist.ProcessGroup | None
	"""

	# Helper to split a qualified name like "a.b.weight" -> ("a.b", "weight")
	def _split_qualified_name(qname: str):
		if "." in qname:
			path, leaf = qname.rsplit(".", 1)
			return path, leaf
		return "", qname

	assert dist.is_initialized(), "torch.distributed must be initialized before calling send_models"

	# 1) Build tensor-free payloads and gather tensors to transfer via GPU p2p
	payloads = []
	per_model_params = []
	per_model_buffers = []
	saved_params = []
	saved_buffers = []

	for model in models:
		# Collect parameter names and tensors
		param_names = []
		param_requires = []
		param_tensors = []
		model_saved_params = []

		for name, p in model.named_parameters(recurse=True):
			param_names.append(name)
			param_requires.append(bool(p.requires_grad))
			pt = p.data
			if not pt.is_cuda:
				pt = pt.to(device="cuda")
			else:
				pt = pt.to(device=torch.cuda.current_device())
			param_tensors.append(pt.contiguous())

			module_path, leaf = _split_qualified_name(name)
			submod = model.get_submodule(module_path) if module_path != "" else model
			# Save original parameter object so we can restore it after pickling
			model_saved_params.append((submod, leaf, p))

			# During pickling (triggered by send_object_list), some module types
			# such as torch.fx.GraphModule may perform symbolic tracing and expect
			# parameters to be non-None. However, we do not want to serialize the
			# full GPU tensors on the CPU channel.
			#
			# To avoid GPU->CPU transfers while still satisfying these expectations,
			# we temporarily replace the parameter with a meta placeholder; the real GPU tensor
			# is sent separately below via p2p.
			dummy_param = nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=False)
			submod._parameters[leaf] = dummy_param  # type: ignore

		# Collect buffer names and tensors
		buffer_names = []
		buffer_tensors = []
		model_saved_buffers = []

		for name, b in model.named_buffers(recurse=True):
			buffer_names.append(name)
			bt = b
			if not bt.is_cuda:
				bt = bt.to(device="cuda")
			else:
				bt = bt.to(device=torch.cuda.current_device())
			buffer_tensors.append(bt.contiguous())

			module_path, leaf = _split_qualified_name(name)
			submod = model.get_submodule(module_path) if module_path != "" else model
			# Save original buffer so we can restore it after pickling
			model_saved_buffers.append((submod, leaf, b))
			# Same rationale as for parameters above: avoid serializing large
			# GPU tensors while keeping non-None buffers for modules that rely
			# on them during pickling.
			dummy_buffer = nn.Parameter(torch.empty_like(b, device="meta"), requires_grad=False)
			submod._buffers[leaf] = dummy_buffer

		payloads.append((model, param_names, param_requires, buffer_names))
		per_model_params.append(param_tensors)
		per_model_buffers.append(buffer_tensors)
		saved_params.append(model_saved_params)
		saved_buffers.append(model_saved_buffers)

	# 2) Send tensor-free module payloads via CPU object communication
	dist.send_object_list(payloads, dst=dst, group=group)

	# 3) Restore original params/buffers in local models
	for model_saved_params in saved_params:
		for submod, leaf, p in model_saved_params:
			submod._parameters[leaf] = p
	for model_saved_buffers in saved_buffers:
		for submod, leaf, b in model_saved_buffers:
			submod._buffers[leaf] = b

	# 4) Send tensor metadata and data via GPU p2p
	for model_params, model_buffers in zip(per_model_params, per_model_buffers):
		for t in model_params:
			meta = TensorMetadata(t).to_tensor()
			dist.send(meta, dst=dst, group=group)
			dist.send(t, dst=dst, group=group)
		for t in model_buffers:
			meta = TensorMetadata(t).to_tensor()
			dist.send(meta, dst=dst, group=group)
			dist.send(t, dst=dst, group=group)


def recv_models(models, src, group=None):
	"""
	Receive a list of models using p2p comms. See :func:`send_models` for more details.

	:param models: list object with correct size, that will be populated with the received models
	:type models: list[Any]
	:param src: source rank
	:type src: int
	:param group: communication group
	:type group: dist.ProcessGroup | None
	"""
	assert dist.is_initialized(), "torch.distributed must be initialized before calling recv_models"

	# Receive tensor-free module payloads
	obj_list = [None for _ in range(len(models))]
	dist.recv_object_list(obj_list, src=src, group=group)

	# Receive tensors via GPU and attach to the received modules
	for i, payload in enumerate(obj_list):
		module, param_names, param_requires, buffer_names = payload

		# Parameters
		for name, req in zip(param_names, param_requires):
			meta_tensor = torch.empty(TensorMetadata.MAX_SIZE, device=torch.cuda.current_device())
			dist.recv(meta_tensor, src=src, group=group)
			meta = TensorMetadata.from_tensor(meta_tensor)
			tensor = torch.empty(tuple(meta.shape), dtype=meta.dtype, device=torch.cuda.current_device())
			dist.recv(tensor, src=src, group=group)

			if "." in name:
				module_path, leaf = name.rsplit(".", 1)
				submod = module.get_submodule(module_path)
			else:
				submod, leaf = module, name
			param_obj = nn.Parameter(tensor, requires_grad=bool(req))
			submod.register_parameter(leaf, param_obj)

		# Buffers
		for name in buffer_names:
			meta_tensor = torch.empty(TensorMetadata.MAX_SIZE, device=torch.cuda.current_device())
			dist.recv(meta_tensor, src=src, group=group)
			meta = TensorMetadata.from_tensor(meta_tensor)
			tensor = torch.empty(tuple(meta.shape), dtype=meta.dtype, device=torch.cuda.current_device())
			dist.recv(tensor, src=src, group=group)

			if "." in name:
				module_path, leaf = name.rsplit(".", 1)
				submod = module.get_submodule(module_path)
			else:
				submod, leaf = module, name
			submod.register_buffer(leaf, tensor)

		module.cuda()
		models[i] = module


def broadcast_models(models, src, group=None):
	"""
	Broadcast a list of models using p2p comms.

	.. warning::
		Efficient broadcasting the same way as :func:`send_models` and :func:`recv_models` is not supported yet. This function may take a lot of time if sending GPU models.
	"""
	dist.broadcast_object_list(models, src=src, group=group)
	for model in models:
		model.cuda()


def prod(iterable):
	if isinstance(iterable, (int, float)):
		return iterable

	acc = 1
	for x in iterable:
		acc *= x

	return acc


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
