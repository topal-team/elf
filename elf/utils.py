"""
Various useful classes / functions
"""

import time
import logging


import torch
import torch.distributed as dist
import torch.nn as nn

from typing import List

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dtypes = [
	torch.float16,
	torch.bfloat16,
	torch.float32,
	torch.float64,
	torch.int16,
	torch.int32,
	torch.int64,
	torch.bool,
]


def _is_mpi():
	return dist.is_initialized() and dist.get_backend() == dist.Backend.MPI


def pretty_print_params(n):
	if n > 1e9:
		return f"{n / 1e9:.1f}B"
	elif n > 1e6:
		return f"{n / 1e6:.1f}M"
	else:
		return f"{int(n)}"


def pretty_print_step(rank, times):
	total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (
		2**30
	)
	memory = torch.cuda.max_memory_allocated() / (2**30)
	info = f"Rank {rank} -\n"
	for k, v in times.items():
		info += f"\t{k} : {v:.2f}s\n"
	info += f"\tPeak memory : {memory:.2f}GB ({100 * memory / total_memory:.2f}%)"
	print(info)


class Placement(List[int]):
	"""
	Wrapper around a list of integers that represents the placement of the pipeline blocks.
	"""

	def __init__(self, placement: List[int]):
		super().__init__(placement)
		assert not dist.is_initialized() or max(self) < dist.get_world_size(), (
			"Placement is out of bounds"
		)

	def get_ids(self, rank: int) -> List[int]:
		"""
		Get the ids of the pipeline blocks that are on the given rank.
		"""
		return [i for i in range(len(self)) if self[i] == rank]

	@staticmethod
	def default(scheduler, pp):
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
		self.end_event.synchronize()
		return self.start_event.elapsed_time(self.end_event) / 1000


def send_models(models, dst, group=None):
	"""
	Sends a list of models using p2p comms
	This method sends the model structure (nn.Module object, parameters metadata, etc) via pickled CPU comm, and the actual tensors via GPU comm.

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
			model_saved_params.append((submod, leaf, p))
			submod._parameters[leaf] = None

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
			model_saved_buffers.append((submod, leaf, b))
			submod._buffers[leaf] = None

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
	Receives a list of models using p2p comms

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
	Broadcasts a list of models using p2p comms
	"""
	dist.broadcast_object_list(models, src, group)
	for m in models:
		m.cuda()
