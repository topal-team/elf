"""
Individual stage computation and communication management
"""

import torch
import torch.distributed as dist
from collections import deque
from .scheduling import OpOptions
from .utils import Timer, TensorMetadata
import logging

logger = logging.getLogger("block")


class Variable:
	"""
	Each block takes multiple variables as input and output, that are either received from or sent to another gpu.
	This object represents one of them.
	"""

	def __init__(self, name, peer, group):
		self.name = name

		self.peer = peer  # block id (not rank!)to receive from / send to
		self.group = group

		self.metadata = None  # shape, dtype
		self.was_metadata_sent = False  # flag to avoid sending metadata twice

		self.waiting = deque()  # (work, tensor) : received object waiting to be processed
		self.kept = deque()  # tensor : object kept for backward (only used for inputs)
		self.finished = deque()  # tensor : object ready to be sent to the peer

	def wait_and_pop(self):
		"""
		Get one element from the waiting queue. Waits for the corresponding communication to complete if needed.
		"""
		work, tensor = self.waiting.popleft()
		if work is not None:
			work.wait()
		return tensor

	def get_buffer(self, size):
		"""
		Get a buffer of the given batch size
		"""
		return self.metadata.get_buffer(size)

	def __str__(self):
		return f"Variable(name={self.name},peer={self.peer},metadata={self.metadata})"

	def __repr__(self):
		return str(self)

	# Debug utility
	def _state(self):
		return f"{str(self)} - waiting: {len(self.waiting)}, kept: {len(self.kept)}, finished: {len(self.finished)}"


class PipelineBlock:
	"""
	Pipelines are made up of sequential blocks, numbered [0..n]
	Each block is one layer or group of contiguous layers placed on one device
	"""

	def __init__(self, model, id_, placement, signature, pp_group, dp_group):
		"""
		:param model: layer / group of layers that will perform the computation
		:type model: nn.Module
		:param id_: number of this block in the pipeline
		:type id_: int
		:param placement: mapping of each id on a device
		:type placement: List[int]
		:param signature: signature of the block
		:type signature: Signature
		"""
		super(PipelineBlock, self).__init__()
		# Block infos
		self.rank = placement[id_]  # global rank
		self.placement = placement  # need it to resolve src/dst later on
		self.model = model.cuda() if torch.cuda.is_available() else model

		self.id = id_  # rank in the model
		self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
		self.is_last = self.id == len(placement) - 1
		self.is_first = self.id == 0

		# Process groups for collective communications
		self.pp_group = pp_group
		self.dp_group = dp_group

		# Helpers to manage data queues and src/dst ranks
		self.signature = signature
		# One input comes from one block
		self.inputs = [
			Variable(name, src, self.pp_group) for name, src in zip(signature.inputs, signature.sources)
		]
		# But an output can be needed by multiple blocks
		self.outputs = [
			[Variable(name, dst, self.pp_group) for dst in dsts]
			for name, dsts in zip(signature.outputs, signature.targets)
		]

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

		# Gather all variables needed for forward
		inputs = []
		for var in self.inputs:
			x = var.wait_and_pop()

			if x.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
				x.requires_grad = True

			inputs.append(x)

		if torch.cuda.is_available():
			torch.cuda.synchronize()
		# Should we synchronize here to make sure all inputs are received?

		with Timer() as timer:
			if options.get(OpOptions.REMAT):
				with torch.no_grad():
					y = self.model(*inputs)
					y = (y,) if not isinstance(y, tuple) else y
			else:
				y = self.model(*inputs)
				y = (y,) if not isinstance(y, tuple) else y
				for var, value in zip(self.outputs, y):
					for dst in var:
						dst.kept.append(value)

		self.compute_time.append(timer.time)

		if torch.cuda.is_available():
			torch.cuda.synchronize()  # Breaks the comp/comm overlap, but without it the results are wrong; TODO: investigate why

		if any(not dst.metadata for var in self.outputs for dst in var):
			self._register_out_metadata(y)

		# Send to next
		for var, value in zip(self.outputs, y):
			for dst in var:
				dst.finished.append(value)

		# Keep inputs for backward
		for var, value in zip(self.inputs, inputs):
			var.kept.append(value)

		if self.is_last:
			output = []
			for var in self.outputs:
				assert len(var) == 1, "Output of the pipeline has multiple destinations"
				output.append(var[0].finished.popleft())
			return output

	def backward(self, **options):
		"""
		Perform the backward pass for one tensor of gradients and register it as computed
		Backward assumes activations AND grads to be on top of the queue

		:param **options: options to modify the backward behaviour
		"""
		logger.debug(f"{self} - Computing one backward with options {options}")

		inputs = []
		for var in self.inputs:
			inputs.append(var.kept.popleft())

		if options.get(OpOptions.REMAT):
			with Timer() as timer:
				act = self.model(*inputs)
			self.compute_time += timer.time
		else:
			act = []
			for var in self.outputs:
				# Even if there are multiple destinations, it's the same tensor for everyone of them
				act.append(var[0].kept.popleft())
				for target in var[1:]:
					target.kept.popleft()

		grads = []
		for var in self.outputs:
			# For one tensor, the gradients are the sum of the gradients from every destination
			g = var[0].wait_and_pop()
			for dst in var[1:]:
				g += dst.wait_and_pop()
			grads.append(g)

		if torch.cuda.is_available():
			torch.cuda.synchronize()
		# Should we synchronize here to make sure all gradients are received?

		with Timer() as timer:
			for i in range(len(self.outputs)):
				if not isinstance(act[i], torch.Tensor):
					continue
				assert (
					act[i].shape == grads[i].shape
				), f"Expected same shape for activations and gradients, got {act[i].shape} and {grads[i].shape}"
				# We may need to keep the graph for multiple backwards, if an intermediate value is needed by next blocks
				act[i].backward(grads[i], retain_graph=(i != len(self.outputs) - 1))

		if torch.cuda.is_available():
			torch.cuda.synchronize()  # Breaks the comp/comm overlap, but without it the results are wrong; TODO: investigate why

		self.compute_time.append(timer.time)

		for var, value in zip(self.inputs, inputs):
			# This is not technically an issue, but it might be a source of bugs
			if not value.requires_grad or value.grad is None:
				# Input usually doesn't need gradients, don't log everytime
				if var.peer is not None:
					logger.warning(f"{self} - No gradient computed for var {var}")
				continue
			var.finished.append(value.grad.data)

	def send_forward(self, **options):
		"""
		Send one activation to the next layer in the model

		:param **options: options to modify the send behaviour
		:return: If the communications need to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		dst = options.get("dst")

		if dst is None or self.placement[dst] == self.rank:
			return

		sends = []
		for var in self.outputs:
			for target in var:
				# only perform communications for that dst
				if target.peer != dst:
					continue

				if not target.was_metadata_sent:
					self._send_metadata(dst)
				outputs = target.finished.popleft().contiguous()

				rank = self.placement[dst]  # we now use the actual rank instead of the block id
				if options.get(OpOptions.BATCHED_COMM):
					sends.append(dist.P2POp(dist.isend, outputs, rank, group=self.pp_group))
				else:
					logger.debug(f"{self} - Sending outputs to rank {rank}")
					dist.isend(outputs, rank, group=self.pp_group)

		return sends  # if not batched, sends is still empty and therefore Falsy

	def send_backward(self, **options):
		"""
		Send one gradient tensor to the previous layer in the model

		:param **options: options to modify the send behaviour
		:return: If the communications need to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		dst = options.get("dst")
		if dst is None or self.placement[dst] == self.rank:
			return

		sends = []
		for var in self.inputs:
			if var.peer != dst:
				continue

			grads = var.finished.popleft().contiguous()

			rank = self.placement[dst]
			if options.get(OpOptions.BATCHED_COMM):
				sends.append(dist.P2POp(dist.isend, grads, rank, group=self.pp_group))
			else:
				logger.debug(f"{self} - Sending gradients to rank {rank}")
				dist.isend(grads, rank, group=self.pp_group)

		return sends  # if not batched, sends is still empty and therefore Falsy

	def recv_forward(self, mb_size, **options):
		"""
		Receive and store one activation to forward

		:param mb_size: size of the micro batch to receive
		:type mb_size: int
		:param **options: options to modify the send behaviour
		:return: If the communications need to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		src = options.get("src")

		# If some metadata is missing, it should be received from the previous block before actual data
		if any(not var.metadata for var in self.inputs):
			self._receive_metadata(src)

		if src is None or self.placement[src] == self.rank:
			return

		recvs = []
		# We couple buffer and work objects for now so that we can wait at the right moment
		for var in self.inputs:
			if var.peer != src:
				continue

			buffer = var.metadata.get_buffer(mb_size)

			rank = self.placement[src]
			if options.get(OpOptions.BATCHED_COMM):
				recvs.append(dist.P2POp(dist.irecv, buffer, rank, group=self.pp_group))
				work = None
			else:
				logger.debug(f"{self} - Starting to receive inputs from rank {rank}")
				work = dist.irecv(buffer, rank, group=self.pp_group)

			var.waiting.append((work, buffer))

		return recvs  # if not batched, recvs is still empty and therefore Falsy

	def recv_backward(self, mb_size, **options):
		"""
		Receive and store one gradient to backward

		:param mb_size: size of the micro batch to receive
		:type mb_size: int
		:param **options: options to modify the send behaviour
		:return: If the communications need to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		src = options.get("src")

		if src is None or self.placement[src] == self.rank:
			return

		recvs = []
		for var in self.outputs:
			for target in var:
				if target.peer != src:
					continue

				buffer = target.metadata.get_buffer(mb_size)

				rank = self.placement[src]
				if options.get(OpOptions.BATCHED_COMM):
					recvs.append(dist.P2POp(dist.irecv, buffer, rank, group=self.pp_group))
					work = None
				else:
					logger.debug(f"{self} - Starting to receive gradients from rank {rank}")
					work = dist.irecv(buffer, rank, group=self.pp_group)

				target.waiting.append((work, buffer))

		return recvs  # if not batched, recvs is still empty and therefore Falsy

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

		for var in self.inputs:
			if src is None or self.placement[src] == self.rank:
				continue
			else:
				if var.peer != src:
					continue

				rank = self.placement[src]  # we resolve the actual rank now
				logger.debug(f"{self} - Receiving metadata from {rank}")
				metadata = torch.empty(TensorMetadata.MAX_SIZE, device=self.device)
				dist.recv(metadata, src=rank, group=self.pp_group)
				var.metadata = TensorMetadata.from_tensor(metadata)

		logger.debug(f"{self} - Registered metadata {[var for var in self.inputs]} from {src}")

	def _send_metadata(self, dst):
		"""
		Send output metadata to the next block. No-op if dst is None or the same as the current rank.

		:param dst: rank of the next block
		:type dst: int
		"""
		if dst is None or self.placement[dst] == self.rank:
			return

		for var in self.outputs:
			for target in var:
				if target.peer != dst:
					continue

				rank = self.placement[dst]  # we resolve the actual rank now
				logger.debug(f"{self} - Sending metadata of {target.name} = {target.metadata} to {rank}")
				dist.send(target.metadata.to_tensor(), dst=rank, group=self.pp_group)
				target.was_metadata_sent = True

	def _register_out_metadata(self, output):
		"""
		Register output metadata from the result of the forward pass.

		:param output: output of the forward pass
		:type output: Tuple[Tensor]
		"""
		for var, value in zip(self.outputs, output):
			for target in var:
				target.metadata = TensorMetadata(value[0])
			logger.debug(f"{self} - registered output metadata {var[0].metadata} for {var[0].name}")
