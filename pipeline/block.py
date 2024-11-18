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

	def __init__(self, model, id_, placement):
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
		# Structure of one element is [[Work, Tensor], [Work, Tensor], ...]
		self.inputs_to_forward = deque()  # Waiting for forward
		self.grads_to_backward = deque()  # Waiting for backward

		# Structure of one element is [Tensor1, Tensor2, ...]
		self.act_to_send = deque()  # Sent to next block
		self.grads_to_send = deque()  # Sent to previous block
		self.act_to_keep = deque()  # Kept for backward
		self.inputs_to_keep = deque()  # Kept for backward

		# Ranks where the previous/next blocks in the model are placed
		self.previous = None if self.id == 0 else placement[self.id - 1]
		self.next = None if self.id == len(placement) - 1 else placement[self.id + 1]

		# Process groups for collective communications
		self.dp_group = None

		self.metadata = []
		self.out_metadata = []

		self.compute_time = []  # used to measure idle time

	def __str__(self) -> str:
		return f"[Layer {self.id} : GPU {self.rank}]"

	def forward(self, **options):
		"""
		Perform the forward pass for one tensor of activations and register it as computed

		:param **options: options to modify the forward behaviour
		:return: if this is the last block of the pipeline, returns its output. Otherwise returns None
		:rtype: Tuple[Tensor] or None
		"""
		logger.debug(f"{self} - Computing one forward with options {options}")

		# Wait for all communications to finish
		inputs = self.inputs_to_forward.popleft()
		for i in range(len(self.metadata)):
			work, x = inputs[i]
			if work is not None:
				work.wait()
			inputs[i] = x

			if x.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
				x.requires_grad = True

		# torch.cuda.synchronize()
		# Should we synchronize here to make sure all inputs are received?

		with Timer() as timer:
			if options.get("remat"):
				with torch.no_grad():
					y = self.model(*inputs)
					y = (y,) if not isinstance(y, tuple) else y
			elif options.get("offload"):
				with activations_offloading():
					y = self.model(*inputs)
					y = (y,) if not isinstance(y, tuple) else y
				self.act_to_keep.append(y)
			else:
				y = self.model(*inputs)
				y = (y,) if not isinstance(y, tuple) else y
				self.act_to_keep.append(y)

		self.compute_time.append(timer.time)

		if torch.cuda.is_available():
			torch.cuda.synchronize() # Breaks the comp/comm overlap, but without it the results are wrong; TODO: investigate why

		if not self.out_metadata:
			self._register_out_metadata(y)

		self.act_to_send.append(y)
		self.inputs_to_keep.append(inputs)

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
				act = self.model(*x)
			self.compute_time += timer.time
		elif options.get("offload"):
			act = self.act_to_keep.popleft().cuda()
		else:
			act = self.act_to_keep.popleft()

		# Wait for all communications to finish
		grads = self.grads_to_backward.popleft()
		for i in range(len(self.out_metadata)):
			work, g = grads[i]
			if work is not None:
				work.wait()
			grads[i] = g

		# torch.cuda.synchronize()
		# Should we synchronize here to make sure all gradients are received?

		with Timer() as timer:
			for i in range(len(self.out_metadata)):
				if not isinstance(act[i], torch.Tensor):
					continue
				assert act[i].shape == grads[i].shape
				# We may need to keep the graph for multiple backwards, if an intermediate value is needed by next blocks
				act[i].backward(grads[i], retain_graph=(i != len(self.out_metadata) - 1))

		if torch.cuda.is_available():
			torch.cuda.synchronize() # Breaks the comp/comm overlap, but without it the results are wrong; TODO: investigate why

		self.compute_time.append(timer.time)

		self.grads_to_send.append(tuple(x[i].grad.data for i in range(len(x)) if x[i].requires_grad))

	def send_forward(self, **options):
		"""
		Send one activation to the next layer in the model

		:param **options: options to modify the send behaviour
		:return: If the communications needs to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		dst = options.get("dst", self.next)

		if dst is None or dst == self.rank:
			return

		# Note: at least one forward was already done, sotorch.cuda.synchronize() output metadata was registered, and sent to next block

		activations = self.act_to_send.popleft()

		if options.get("batched_comm"):
			sends = []
			for i in range(len(self.out_metadata)):
				tensor = activations[i].contiguous()
				sends.append(dist.P2POp(dist.isend, tensor, dst, group=self.pp_group))
			return sends
		else:
			for i in range(len(self.out_metadata)):
				logger.debug(f"{self} - Sending activation to layer {self.id + 1} on rank {dst} (shape = {activations[i].shape})")
				tensor = activations[i].contiguous()
				dist.isend(tensor, dst, group=self.pp_group)

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
			sends = []
			for i in range(len(self.metadata)):
				tensor = grads[i].contiguous()
				sends.append(dist.P2POp(dist.isend, tensor, dst, group=self.pp_group))
			return sends
		else:
			logger.debug(f"{self} - Sending gradients to layer {self.id - 1} on rank {dst}")
			for i in range(len(self.metadata)):
				tensor = grads[i].contiguous()
				dist.isend(tensor, dst, group=self.pp_group)

	def recv_forward(self, mb_size, **options):
		"""
		Receive and store one activation to forward

		:param mb_size: size of the micro batch to receive
		:type mb_size: int
		:param **options: options to modify the send behaviour
		:return: If the communications needs to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		src = options.get("src", self.previous)

		if options.get("offload") and len(self.act_to_keep) > 0:
			# Free memory just before allocating the next buffer
			activations_offloading().wait_for_offloading()

		if not self.metadata:
			self._receive_metadata(src)

		if src is None or src == self.rank:
			return

		buffers = []
		# We couple buffer and work objects for now so that we can wait at the right moment
		for i in range(len(self.metadata)):
			buffers.append([None, self.metadata[i].get_buffer(mb_size)])

		self.inputs_to_forward.append(buffers)

		if options.get("batched_comm"):
			# This communication needs to be batched ;
			# instead of executing it, we instanciate an object with the right setup and return it
			recvs = []
			for i in range(len(self.metadata)):
				recvs.append(dist.P2POp(dist.irecv, buffers[i][1], src, group=self.pp_group))
			return recvs

		else:
			for i in range(len(self.metadata)):
				logger.debug(
					f"{self} - Starting to receive activations with shape {self.metadata} from layer {self.id - 1} on rank {src} (shape = {buffers[i][1].shape})"
				)
				work = dist.irecv(buffers[i][1], src, group=self.pp_group)
				buffers[i][0] = work

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

		buffers = []
		for i in range(len(self.out_metadata)):
			buffers.append([None, self.out_metadata[i].get_buffer(mb_size)])

		self.grads_to_backward.append(buffers)

		if options.get("batched_comm"):
			# This communication needs to be batched ;
			# instead of executing it, we instanciate an object with the right setup and return it
			recvs = []
			for i in range(len(self.out_metadata)):
				recvs.append(dist.P2POp(dist.irecv, buffers[i][1], src, group=self.pp_group))
			return recvs

		else:	
			logger.debug(
				f"{self} - Starting to receive gradients with shape {self.out_metadata} from layer {self.id + 1} on rank {src}"
			)
			for i in range(len(self.out_metadata)):
				work = dist.irecv(buffers[i][1], src, group=self.pp_group)
				buffers[i][0] = work

	def all_reduce_param_grads(self, **options):
		"""
		All-reduce operation on the gradients of the model parameters across the data parallel group.
		:param **options: Additional options to modify the all-reduce behaviour.
		:type **options: dict
		"""
		if self.dp_group is None:
			return
		for _, p in sorted(self.model.named_parameters()):
			dist.all_reduce(p.grad.data, group=self.dp_group, op=dist.ReduceOp.AVG)

	def scale_grads(self, batch_size):
		"""
		Scale the gradients of the model parameters by the batch size

		:param batch_size: size of the full batch
		:type batch_size: int
		"""
		for p in self.model.parameters():
			if p.requires_grad:
				p.grad.data /= batch_size

	def _receive_metadata(self, src):
		"""
		Register input metadata by receiving it from the previous block, or using the current input tensor if src is None or the same as the current rank.

		:param src: rank of the previous block
		:type src: int
		"""
		if src is None or src == self.rank:
			x = self.inputs_to_forward[0]
			n = len(x)
			for i in range(len(x)):
				_, tensor = x[i]
				self.metadata.append(TensorMetadata(tensor[0]))
		else:
			x = torch.empty(1, device=self.device, dtype=torch.int32)
			dist.recv(x, src=src, group=self.pp_group)
			n = int(x.item())
			for _ in range(n):
				metadata = torch.empty(TensorMetadata.MAX_SIZE, device=self.device)
				dist.recv(metadata, src=src, group=self.pp_group)
				self.metadata.append(TensorMetadata.from_tensor(metadata))

		logger.debug(f"{self} - Registered metadata {self.metadata}")
		logger.debug(f"{self} - has {n} inputs")

	def _send_metadata(self, dst):
		"""
		Send output metadata to the next block. No-op if dst is None or the same as the current rank.

		:param dst: rank of the next block
		:type dst: int
		"""
		if dst is None or dst == self.rank:
			return

		# Send number of outputs
		n = torch.empty(1, device=self.device, dtype=torch.int32)
		n[0] = len(self.out_metadata)
		dist.send(n, dst=dst, group=self.pp_group)
		logger.debug(f"{self} - sent number of outputs ({n.item()}) to {dst}")

		# Send metadata
		for i in range(len(self.out_metadata)):
			dist.send(self.out_metadata[i].to_tensor(), dst=dst, group=self.pp_group)
			logger.debug(f"{self} - sent metadata {self.out_metadata[i]}")

	def _register_out_metadata(self, output):
		"""
		Register output metadata from the result of the forward pass. Then sends it to the next block.

		:param output: output of the forward pass
		:type output: Tuple[Tensor]
		"""
		for o in output:
			self.out_metadata.append(TensorMetadata(o[0]))
			logger.debug(f"{self} - registered output metadata {self.out_metadata[-1]}")

		self._send_metadata(self.next)  # not very elegant to do this here as we don't have dst
