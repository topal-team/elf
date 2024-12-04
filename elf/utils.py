"""
Various useful classes / functions
"""

import time
import uuid
import torch
import torch.distributed as dist

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


def op_to_str(op):
	"""
	Pretty print for dist.P2POp

	:param op: communication operation
	:type op: dist.P2POp
	:return: string describing the op
	:rtype: string
	"""
	match op.op:
		case dist.isend:
			return f"Send to {op.peer}"
		case dist.irecv:
			return f"Receive from {op.peer}"


def pretty_print_params(n):
	if n > 1e9:
		return f"{n/1e9:.1f}B"
	elif n > 1e6:
		return f"{n/1e6:.1f}M"
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
	Utility context to time the execution of some code
	Uses GPU if cuda is available, else times CPU execution

	Utilisation example: ::

		with Timer() as timer:
			# do some stuff
			# everything in this context will be timed
		# if you do something here, it will not be timed
		print(timer.time())

	"""

	def __new__(cls, *args):
		if args:
			if args[0].lower() in ["gpu", "cuda"]:
				return TimerGPU()
			elif args[0].lower() == "cpu":
				return TimerCPU()
		if torch.cuda.is_available():
			return TimerGPU()
		else:
			return TimerCPU()


class TimerCPU:
	"""
	Timer for CPU execution
	"""

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

	def __init__(self):
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


class activations_offloading(torch.autograd.graph.saved_tensors_hooks):
	"""
	Context to offload activations on CPU

	Usage: ::

		with activations_offloading():
			y = model(x)
		# do stuff
		activations_offloading().wait_for_offloading() # GPU memory for activations is freed
		# ...
		activations_offloading().prefetch() # GPU memory is re-allocated and activations moved back to CPU
		# ...
		loss.backward() # all tensors are on GPU for backward

	.. warning::
		Activation offloading has some known issues that cause CPU memory to be overused.

	"""

	# Singleton pattern
	_instance = None

	def __new__(class_):
		if not isinstance(class_._instance, class_):
			class_._instance = object.__new__(class_)
			class_._instance.events = {}
			class_._instance.tensors = {}
			class_._instance.stream = torch.cuda.Stream()
			class_._instance.single = False
		return class_._instance

	def __init__(self):
		if self.single:
			return
		self.single = True

		# When forward is done, we start sending to cpu asynchronously
		def pack_to_cpu(tensor):
			key = uuid.uuid4()
			self.events[key] = torch.cuda.Event()
			packed = torch.empty(tensor.size(), device=torch.device("cpu"), dtype=tensor.dtype)
			with torch.cuda.stream(self.stream):
				packed.copy_(tensor, non_blocking=True)
				self.tensors[key] = packed
				self.events[key].record(self.stream)
				del tensor, packed

			return key

		# Ensure the data movement was finished and return the device tensor
		def unpack_from_cpu(key):
			# If it wasn't prefetched just copy it
			if not self.events.get(key):
				print("Tensor was not prefetched :/")
				return None

			self.events[key].synchronize()
			unpacked = self.tensors[key]
			del self.events[key]
			del self.tensors[key]
			return unpacked

		super().__init__(pack_to_cpu, unpack_from_cpu)

	# At some point we need to free memory ; this means potentially waiting for the copy to finish, so we want to do it just in time, not too early
	def wait_for_offloading(self):
		"""
		Wait for every activation to be completely moved to CPU, which allows CUDA to free their memory
		"""
		for key in list(self.events.keys()):
			self.events[key].synchronize()
			self.total_size += self.tensors[key].numel() * self.tensors[key].element_size()
		# Here all tensors are effectively copied on cpu and memory on gpu is freed

	# Copy back from device to host, asynchronously
	def prefetch(self):
		for key in list(self.events.keys()):
			self.events[key] = torch.cuda.Event()
			tensor = self.tensors[key]
			unpacked = torch.empty(tensor.size(), dtype=tensor.dtype, device=torch.cuda.current_device())
			with torch.cuda.stream(self.stream):
				unpacked.copy_(tensor, non_blocking=True)
			self.tensors[key] = unpacked
			self.events[key].record(self.stream)


# TODO: instead of sending full tensors, we could send parameter/buffers metadata only
# and then construct tensors on the recv side directly


def send_models(models, dst, group=None):
	"""
	Sends a list of models using p2p comms
	"""
	# Send using CPU
	dist.send_object_list(models, dst, group)


def recv_models(models, src, group=None):
	"""
	Receives a list of models using p2p comms
	"""
	dist.recv_object_list(models, src, group)
	for m in models:
		m.cuda()


def broadcast_models(models, src, group=None):
	"""
	Broadcasts a list of models using p2p comms
	"""
	dist.broadcast_object_list(models, src, group)
	for m in models:
		m.cuda()

	return models
