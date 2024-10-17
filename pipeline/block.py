"""
Individual stage computation and communication management
"""

import torch
import torch.distributed as dist
from collections import deque
from .utils import Timer, TensorMetadata, activations_offloading
import logging

logger = logging.getLogger("block")


class PipelineBlock:
	"""
	Pipelines are made up of sequential blocks, numbered [0..n]
	Each block is one layer or group of contiguous layers placed on one device
	"""

	def __init__(self, model, id_, placement, inputs, outputs):
		"""
		:param model: layer / group of layers that will perform the computation
		:type model: nn.Module
		:param id_: number of this block in the pipeline
		:type id_: int
		:param placement: mapping of each id on a device
		:type placement: List[int]
		:param inputs: name of variables taken as input by the block
		:type inputs: List[str]
		:param outputs: name of variables returned by the block
		:type outputs: List[str]
		"""
		super(PipelineBlock, self).__init__()
		# Block infos
		self.rank = placement[id_]  # global rank
		self.model = model.cuda() if torch.cuda.is_available() else model

		self.id = id_  # rank in the model
		self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

		# Queues of tensors to process
		# Structure of one element is {variable1: [Work, Tensor], variable2: [Work, Tensor], ..}
		self.inputs_to_forward = deque()  # Waiting for forward
		self.grads_to_backward = deque()  # Waiting for backward

		# Structure of one element is {variable1: Tensor, variable2: Tensor, ..}
		self.act_to_send = deque()  # Sent to next block
		self.grads_to_send = deque()  # Sent to previous block
		self.act_to_keep = deque()  # Kept for backward
		self.inputs_to_keep = deque()  # Kept for backward

		# Ranks where the previous/next blocks in the model are placed
		self.previous = None if self.id == 0 else placement[self.id - 1]
		self.next = None if self.id == len(placement) - 1 else placement[self.id + 1]

		# Process groups for collective communications
		self.dp_group = None

		self.metadata = {}
		self.out_metadata = {}

		self.compute_time = 0  # used to measure idle time

		self.inputs = sorted(inputs)  # name of input variables
		self.outputs = sorted(outputs)  # name of output variables
		# sorted alphabetically to make sure the order is consistent across devices
		# note: can this order matter ? can it be faster to communicate in some order depending on the tensor shapes/sizes ?

	def __str__(self) -> str:
		return f"[Layer {self.id} : GPU {self.rank}]"

	def forward(self, **options):
		"""
		Perform the forward pass for one tensor of activations and register it as computed

		:param **options: options to modify the forward behaviour
		:return: if this is the last block of the pipeline, returns its output. Otherwise returns None
		:rtype: Tensor or None
		"""
		logger.debug(f"{self} - Computing one forward with options {options}")

		# Wait for all communications to finish
		x = self.inputs_to_forward.popleft()
		for key in self.inputs:
			work, i = x[key]
			if work is not None:
				work.wait()
			x[key] = i

			if x[key].dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
				x[key].requires_grad = True

		with Timer() as timer:
			if options.get("remat"):
				with torch.no_grad():
					y = self.model(**x)
			elif options.get("offload"):
				with activations_offloading():
					y = self.model(**x)
				self.act_to_keep.append(y)
				# x = x.cpu()
			else:
				y = self.model(**x)
				self.act_to_keep.append(y)

		self.compute_time += timer.time()

		self.act_to_send.append(y)
		self.inputs_to_keep.append(x)

		if self.next is None:
			return self.act_to_send.popleft()

	def backward(self, **options):
		"""
		Perform the backward pass for one tensor of gradients and register it as computed
		Backward assumes activations AND grads to be on top of the queue

		:param **options: options to modify the backward behaviour
		"""
		logger.debug(f"{self} - Computing one backward with options {options}")

		x = self.inputs_to_keep.popleft()

		if options.get("remat"):
			with Timer() as timer:
				act = self.model(**x)
			self.compute_time += timer.time()
		elif options.get("offload"):
			act = self.act_to_keep.popleft().cuda()
		else:
			act = self.act_to_keep.popleft()

		# Wait for all communications to finish
		grads = self.grads_to_backward.popleft()
		for key in self.outputs:
			work, g = grads[key]
			if work is not None:
				work.wait()
			grads[key] = g

		with Timer() as timer:
			for key in self.outputs:
				# Perform a backward pass for each output tensor; once the last one is done, the graph can be freed
				act[key].backward(grads[key], retain_graph=(key != self.outputs[-1]))
		self.compute_time += timer.time()

		self.grads_to_send.append(
			{key: value.grad.data for key, value in x.items() if value.requires_grad}
		)

	def send_forward(self, **options):
		"""
		Send one activation to the next layer in the model

		:param **options: options to modify the send behaviour
		:return: If the communications needs to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		dst = options.get("dst") or self.next
		if not self.out_metadata:
			self._send_metadata(dst)

		if dst is None or dst == self.rank:
			return

		activations = self.act_to_send.popleft()

		if options.get("batched_comm"):
			return [dist.P2POp(dist.isend, activations[out], dst, group=self.pp_group) for out in self.outputs]
		else:
			logger.debug(f"{self} - Sending activations to layer {self.id + 1} on rank {dst}")
			for out in self.outputs:
				dist.isend(activations[out], dst, group=self.pp_group)

	def send_backward(self, **options):
		"""
		Send one gradient tensor to the previous layer in the model

		:param **options: options to modify the send behaviour
		:return: If the communications needs to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		dst = options.get("dst") or self.previous
		if dst is None or dst == self.rank:
			return

		grads = self.grads_to_send.popleft()

		if options.get("batched_comm"):
			return [dist.P2POp(dist.isend, grads[inp], dst, group=self.pp_group) for inp in self.inputs]
		else:
			logger.debug(f"{self} - Sending gradients to layer {self.id - 1} on rank {dst}")
			for inp in self.inputs:
				dist.isend(grads[inp], dst, group=self.pp_group)

	def recv_forward(self, mb_size, **options):
		"""
		Receive and store one activation to forward

		:param mb_size: size of the micro batch to receive
		:type mb_size: int
		:param **options: options to modify the send behaviour
		:return: If the communications needs to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		src = options.get("src") or self.previous

		if not self.metadata:
			self._receive_metadata(src)

		if options.get("offload") and len(self.act_to_keep) > 0:
			# Free memory just before allocating the next buffer
			activations_offloading().wait_for_offloading()

		if src is None or src == self.rank:
			return

		buffers = {}
		for key in self.inputs:
			buffers[key] = [None, self.metadata[key].get_buffer(mb_size)]

		if options.get("batched_comm"):
			# This communication needs to be batched ;
			# instead of executing it, we instanciate an object with the right setup and return it
			self.inputs_to_forward.append(buffers)
			return [
				dist.P2POp(dist.irecv, buffers[key][1], src, group=self.pp_group) for key in self.inputs
			]

		else:
			logger.debug(
				f"{self} - Starting to receive activations with shape {self.metadata} from layer {self.id - 1} on rank {src}"
			)
			stream = torch.cuda.Stream()
			with torch.cuda.stream(stream):
				for key in self.inputs:
					work = dist.irecv(buffers[key][1], src, group=self.pp_group)
					buffers[key][0] = work

			torch.cuda.current_stream().wait_stream(stream)  # needed ?
			self.inputs_to_forward.append(buffers)

	def recv_backward(self, mb_size, **options):
		"""
		Receive and store one gradient to backward

		:param mb_size: size of the micro batch to receive
		:type mb_size: int
		:param **options: options to modify the send behaviour
		:return: If the communications needs to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		src = options.get("src") or self.next

		if options.get("offload"):
			# Start moving activations back to gpu
			activations_offloading().prefetch()

		if src is None or src == self.rank:
			return

		buffers = {}
		for key in self.outputs:
			buffers[key] = [None, self.out_metadata[key].get_buffer(mb_size)]

		if options.get("batched_comm"):
			# This communication needs to be batched ;
			# instead of executing it, we instanciate an object with the right setup and return it
			self.grads_to_backward.append(buffers)
			return [
				dist.P2POp(dist.irecv, buffers[key][1], src, group=self.pp_group) for key in self.outputs
			]

		else:
			logger.debug(
				f"{self} - Starting to receive gradients with shape {self.out_metadata} from layer {self.id + 1} on rank {src}"
			)
			stream = torch.cuda.Stream()
			with torch.cuda.stream(stream):
				for key in self.outputs:
					work = dist.irecv(buffers[key][1], src, group=self.pp_group)
					buffers[key][0] = work

			torch.cuda.current_stream().wait_stream(stream)  # needed ?
			self.grads_to_backward.append(buffers)

	def all_reduce_param_grads(self, **options):
		"""
		All-reduce operation on the gradients of the model parameters across the data parallel group.
		:param **options: Additional options to modify the all-reduce behaviour.
		:type **options: dict
		"""
		if self.dp_group is None:
			return
		for _, p in sorted(self.model.named_parameters()):
			dist.all_reduce(p.grad.data, group=self.dp_group)

	def scale_grads(self, batch_size):
		"""
		Scale the gradients of the model parameters by the batch size

		:param batch_size: size of the full batch
		:type batch_size: int
		"""
		for p in self.model.parameters():
			p.grad.data /= batch_size

	def _receive_metadata(self, src):
		for key in self.inputs:
			if src is None or src == self.rank:
				return
				# assert len(self.inputs_to_forward) != 0, "Can't register metadata without inputs"
				# inputs = self.inputs_to_forward[0]  # First mb
				# _, x = inputs[key]  # Correct variable, discard fake work
				# self.metadata[key] = TensorMetadata(x[0])  # Don't register batch size
			else:
				metadata = torch.empty(TensorMetadata.MAX_SIZE, device=self.device)
				dist.recv(metadata, src=src)
				self.metadata[key] = TensorMetadata.from_tensor(metadata)
			logger.debug(f"{self} - Registered metadata {self.metadata}")
	
	def _send_metadata(self, dst):
		if dst is None or dst == self.rank:
			return
		assert len(self.act_to_send) != 0, "Can't send metadata without activations"
		y = self.act_to_send[0]

		for k in self.outputs:
			if not isinstance(y[k], torch.Tensor):
				raise RuntimeError(f"Non-tensor output from block {self} : key {k} has type {type(y[k])}.")
			self.out_metadata[k] = TensorMetadata(y[k][0])
			logger.debug(f"{self} - Registered out-metadata {self.out_metadata}")
			if dst is not None and dst != self.rank:
				dist.send(self.out_metadata[k].to_tensor(), dst=dst)