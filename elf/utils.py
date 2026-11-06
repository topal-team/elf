"""
Various useful classes / functions.

Everything here is independent of the rest of the library: it imports nothing from ``elf`` and
knows nothing about pipelines.
"""

import time
import logging

from contextlib import contextmanager

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

__all__ = [
	"TensorMetadata",
	"dtypes",
	"Timer",
	"TimerCPU",
	"TimerGPU",
	"NameMapping",
	"prod",
	"pretty_print_params",
	"tensor_leaves",
	"rng_state",
	"restored_rng",
]


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


def prod(iterable):
	if isinstance(iterable, (int, float)):
		return iterable

	acc = 1
	for x in iterable:
		acc *= x

	return acc


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


# === Code taken from Rotor (https://gitlab.inria.fr/hiepacs/rotor)


def tensor_leaves(args):
	"""
	Walk a possibly nested structure and yield every tensor it holds.

	Dictionaries are walked in insertion order, so that a structure and a copy of it rebuilt from
	the same walk line up leaf for leaf.

	:param args: tensor, or a nesting of tuples, lists and dicts of them
	:return: the tensors, in order
	:rtype: Iterator[torch.Tensor]
	"""
	if isinstance(args, torch.Tensor):
		yield args
	elif isinstance(args, (tuple, list)):
		for arg in args:
			yield from tensor_leaves(arg)
	elif isinstance(args, dict):
		for arg in args.values():
			yield from tensor_leaves(arg)


def rng_state(args):
	"""
	Snapshot the generators a forward is about to draw from.

	:param args: arguments of that forward, used to find which devices are involved
	:type args: tuple
	:return: the cpu generator state, and the state of every cuda device the arguments live on
	:rtype: tuple[torch.Tensor, dict[int, torch.Tensor]]
	"""
	devices = {tensor.device.index for tensor in tensor_leaves(args) if tensor.is_cuda}
	return torch.get_rng_state(), {device: torch.cuda.get_rng_state(device) for device in devices}


@contextmanager
def restored_rng(state):
	"""
	Put the generators back where :func:`rng_state` found them, for the duration of a
	recomputation, and leave them where they were for whatever runs next.

	:param state: a state returned by :func:`rng_state`, or None to draw as usual
	:type state: tuple or None
	"""
	if state is None:
		yield
		return

	cpu, cuda = state
	with torch.random.fork_rng(devices=list(cuda), enabled=True):
		torch.set_rng_state(cpu)
		for device, value in cuda.items():
			torch.cuda.set_rng_state(value, device)
		yield


def ensure_tuple(x):
	return (x,) if not isinstance(x, (tuple, list)) else x


def get_device(args):
	"""
	Returns the device of the first tensor found in a possibly nested structure
	"""
	if isinstance(args, torch.Tensor):
		return args.device
	elif hasattr(args, "__iter__"):
		for arg in args:
			device = get_device(arg)
			if device:
				return device

	return None


# Hide is used on the output of a forward, so that we do not have
# to keep the output and still be able to do backward() on them.
class HideFun(torch.autograd.Function):
	@staticmethod
	def forward(ctx, grad_store, *inputs):
		ctx.grad_store = grad_store
		return torch.zeros(1, device=get_device(inputs))

	@staticmethod
	def backward(ctx, _):
		grad = ctx.grad_store[0]  # incoming gradient
		ctx.grad_store.clear()  # clear the list without deleting it in case there are multiple backwards
		return (None, *grad)  # None for grad_store


class Hide:
	def __init__(self, t):
		self.storage = []
		self.v = HideFun.apply(self.storage, *ensure_tuple(t))

	def backward(self, gradient=None, *args, **kwargs):
		self.storage.append(ensure_tuple(gradient))
		self.v.backward(*args, **kwargs)
		# storage was cleared by HideFun.backward

	@property
	def grad_fn(self):
		return self.v.grad_fn


# PreHide and SingleUse are used to be able to forget the input to a function
# And still be able to recover the gradient. We provide grad_store where the
# gradient will be captured during backward and stored so that we can recover it
# If any reference is kept to the actual input, it is not by us.
# real_tensor_store is a list so that autograd does not see it as Tensor
class PreHideFun(torch.autograd.Function):
	@staticmethod
	def forward(ctx, real_tensor_store, grad_store, arg):
		ctx.grad_store = grad_store
		ctx.original_device = get_device(arg)
		return real_tensor_store[0]

	@staticmethod
	def backward(ctx, grad):
		ctx.grad_store.append(grad)
		return None, None, torch.zeros(1, device=ctx.original_device)


class SingleUse:
	def __init__(self, object):
		self.content = object

	def get(self):
		result = self.content
		del self.content
		return result


class PreHide:
	def __init__(self, t):
		self.storage = []
		self.requires_grad = t.requires_grad
		self.source = torch.zeros(1, requires_grad=self.requires_grad, device=get_device(t))
		self.v = SingleUse(PreHideFun.apply([t], self.storage, self.source))

	def get(self):
		return self.v.get()

	@property
	def grad(self):
		if self.storage:
			return self.storage[0]
		return None


"""
input = PreHide(input)
result = func(input.get())
hidden_result = Hide(result)
"""
