"""
Individual stage computation and communication management
"""

import torch
import torch.distributed as dist
from .scheduling import OpOptions
from .utils import Timer, TensorMetadata
from .zb_utils import LayerDW
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

		# Single-use data structures ; data is deleted after being read
		self.to_process = []  # (work, tensor) : received object waiting to be processed
		self.saved = []  # tensor : object kept for backward
		self.to_send = []  # tensor : object ready to be sent to the peer

	def get(self, queue, mb_id):
		"""
		Get an element from the specified queue at the given micro-batch index.

		:param queue: Queue to get data from (to_process, saved, or to_send)
		:type queue: List
		:param mb_id: Micro-batch index to retrieve
		:type mb_id: int
		:return: Data at the specified index
		:rtype: torch.Tensor or Any
		:raises Exception: If queue is shorter than mb_id or if data at mb_id is None
		"""
		if len(queue) <= mb_id:
			raise Exception(f"Variable {self.name} - Data queue shorter than mb_id {mb_id}")
		if queue[mb_id] is None:
			raise Exception(f"Variable {self.name} - Trying to pop data at mb_id {mb_id}, but it's empty")

		data = queue[mb_id]
		queue[mb_id] = None
		return data

	def set(self, queue, mb_id, value):
		"""
		Set an element in the specified queue at the given micro-batch index.

		:param queue: Queue to set data in (to_process, saved, or to_send)
		:type queue: List
		:param mb_id: Micro-batch index to set
		:type mb_id: int
		:param value: Data to set
		:type value: torch.Tensor or Any
		:raises Exception: If queue is shorter than mb_id or if data at mb_id is already set
		"""
		if len(queue) <= mb_id:
			queue.extend([None] * (mb_id - len(queue) + 1))
		if queue[mb_id] is not None:
			raise Exception(
				f"Variable {self.name} - Trying to set data at mb_id {mb_id}, but it's already set"
			)
		queue[mb_id] = value

	def wait_and_pop(self, mb_id):
		"""
		Get one element from the waiting queue. Waits for the corresponding communication to complete if needed.

		:param mb_id: Micro-batch index to retrieve
		:type mb_id: int
		:return: Data at the specified index
		:rtype: torch.Tensor or Any
		"""
		work, tensor = self.get(self.to_process, mb_id)
		if work is not None:
			work.wait()
		return (
			tensor.detach()
		)  # communications are differentiable, we detach to delete any unnecessary node

	def get_buffer(self, size):
		"""
		Get a buffer of the given batch size

		:param size: Size of the batch
		:type size: int
		:return: Buffer of the given size
		:rtype: torch.Tensor
		"""
		return self.metadata.get_buffer(size)

	def __str__(self):
		return f"Variable(name={self.name},peer={self.peer},metadata={self.metadata})"

	def __repr__(self):
		return str(self)

	def clear(self):
		"""
		Clear all queues for this variable.
		"""
		self.to_process.clear()
		self.saved.clear()
		self.to_send.clear()

	# Debug utility
	def _state(self):
		"""
		Debug utility to print the state of the variable.
		"""
		to_process = len([x for x in self.to_process if x is not None])
		saved = len([x for x in self.saved if x is not None])
		to_send = len([x for x in self.to_send if x is not None])
		return f"{str(self)} - to process: {to_process}, saved: {saved}, to send: {to_send}"


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

		# One input comes from one block (List[Variable])
		self.input_variables = [
			Variable(name, src, self.pp_group) for name, src in zip(signature.inputs, signature.sources)
		]
		# But an output can be needed by multiple blocks (List[List[Variable]])
		self.output_variables = [
			[Variable(name, dst, self.pp_group) for dst in dsts]
			for name, dsts in zip(signature.outputs, signature.targets)
		]

		self.compute_time = []  # used to measure idle time

	def __str__(self) -> str:
		return f"[Layer {self.id} : GPU {self.rank}]"

	def forward(self, mb_id, **options):
		"""
		Perform the forward pass for one tensor of activations and register it as computed

		:param **options: options to modify the forward behaviour
		:return: if this is the last block of the pipeline, returns its output. Otherwise returns None
		:rtype: Tuple[Tensor] or None
		"""
		logger.debug(f"{self} - Computing one forward with options {options}")

		# Gather all variables needed for forward
		inputs = []
		for input_var in self.input_variables:
			x = input_var.wait_and_pop(mb_id)

			if x.is_floating_point():
				x.requires_grad_(True)

			inputs.append(x)

		with Timer(name=f"forward({self.id}:{mb_id})") as timer:
			if options.get(OpOptions.REMAT):
				with torch.no_grad():
					y = self.model(*inputs)
					y = (y,) if not isinstance(y, tuple) else y

					# We'll need to recompute the forward pass, so we keep the inputs in the waiting queue
					for input_var, value in zip(self.input_variables, inputs):
						input_var.set(input_var.to_process, mb_id, (None, value))
			else:
				y = self.model(*inputs)
				y = (y,) if not isinstance(y, tuple) else y

				# Keep both outputs and inputs for backward
				for output_var, value in zip(self.output_variables, y):
					for output_dst in output_var:
						output_dst.set(output_dst.saved, mb_id, value)

				for input_var, value in zip(self.input_variables, inputs):
					input_var.set(input_var.saved, mb_id, value)

		self.compute_time.append(timer)

		if any(not dst.metadata for var in self.output_variables for dst in var):
			self._register_out_metadata(y)

		# Register what will be sent to next
		for output_var, value in zip(self.output_variables, y):
			for output_dst in output_var:
				output_dst.set(output_dst.to_send, mb_id, value.detach())

		if self.is_last:
			output = []
			for output_var in self.output_variables:
				assert len(output_var) == 1, "Output of the pipeline has multiple destinations"
				output.append(output_var[0].get(output_var[0].to_send, mb_id))
			return output

	def backward_inputs(self, mb_id, **options):
		"""
		Perform the backward pass for one tensor of gradients and register it as computed
		Backward assumes activations AND grads to be on top of the queue

		:param **options: options to modify the backward behaviour
		"""
		logger.debug(f"{self} - Computing one backward with options {options}")

		# Gather the inputs to access their gradients later on
		inputs = []
		for input_var in self.input_variables:
			inputs.append(input_var.get(input_var.saved, mb_id))

		# Gather the outputs
		outputs = []
		for output_var in self.output_variables:
			# Even if there are multiple destinations, it's the same tensor for all of them
			# retrieve the first one, delete others
			outputs.append(output_var[0].get(output_var[0].saved, mb_id))
			for output_dst in output_var[1:]:
				output_dst.get(output_dst.saved, mb_id)

		for i, output_var in enumerate(self.output_variables):
			# wait for all recvs to be finished for this variable
			incoming_grads = []
			for output_dst in output_var:
				incoming_grads.append(output_dst.wait_and_pop(mb_id))

			with Timer(name=f"backward({self.id}:{mb_id})") as timer:
				# sum up all the gradients from all destinations
				gradient_accumulator = incoming_grads[0]
				for grad in incoming_grads[1:]:
					gradient_accumulator += grad

				if not isinstance(outputs[i], torch.Tensor):
					continue
				assert outputs[i].shape == gradient_accumulator.shape, (
					f"Expected same shape for activations and gradients, got {outputs[i].shape} and {gradient_accumulator.shape}"
				)
				# We may need to keep the graph for multiple backwards, if an intermediate value is needed by next blocks
				outputs[i].backward(
					gradient_accumulator, retain_graph=(i != len(self.output_variables) - 1)
				)

		self.compute_time.append(timer)

		for input_var, value in zip(self.input_variables, inputs):
			# This is not technically an issue, but it might be a source of bugs
			if not value.requires_grad or value.grad is None:
				# Input usually doesn't need gradients, don't log everytime
				if input_var.peer is not None:
					logger.warning(f"{self} - No gradient computed for var {input_var}")
				continue
			input_var.set(input_var.to_send, mb_id, value.grad.data)

	def backward_params(self, mb_id, **options):
		"""
		Perform the backward pass for the parameters of the model

		:param mb_id: micro-batch index
		:type mb_id: int
		:param **options: options to modify the backward behaviour
		"""
		with Timer(name=f"backward_params({self.id}:{mb_id})") as timer:
			for name, module in self.model.named_modules():
				if isinstance(module, LayerDW):
					logger.debug(f"{self} - Backwarding params of {name}")
					module.backward()
				else:
					logger.debug(f"{self} - {name} is not a LayerDW ; skipping")
		self.compute_time.append(timer)

	def send_forward(self, mb_id, **options):
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
		for var in self.output_variables:
			for target in var:
				# only perform communications for that dst
				if target.peer != dst:
					continue

				if not target.was_metadata_sent:
					self._send_metadata(dst)
				outputs = target.get(target.to_send, mb_id).contiguous()

				rank = self.placement[dst]  # we now use the actual rank instead of the block id
				if options.get(OpOptions.BATCHED_COMM):
					sends.append(dist.P2POp(dist.isend, outputs, rank, group=self.pp_group))
				else:
					logger.debug(f"{self} - Sending outputs to rank {rank}")
					dist.isend(outputs, rank, group=self.pp_group)

		return sends  # if not batched, sends is still empty and therefore Falsy

	def send_backward(self, mb_id, **options):
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
		for var in self.input_variables:
			if var.peer != dst:
				continue

			grads = var.get(var.to_send, mb_id).contiguous()

			rank = self.placement[dst]
			if options.get(OpOptions.BATCHED_COMM):
				sends.append(dist.P2POp(dist.isend, grads, rank, group=self.pp_group))
			else:
				logger.debug(f"{self} - Sending gradients to rank {rank}")
				dist.isend(grads, rank, group=self.pp_group)

		return sends  # if not batched, sends is still empty and therefore Falsy

	def recv_forward(self, mb_id, mb_size, **options):
		"""
		Receive and store one activation to forward

		:param mb_size: size of the micro batch to receive
		:type mb_size: int
		:param **options: options to modify the send behaviour
		:return: If the communications need to be batched, returns them
		:rtype: List[dist.P2POp] or None
		"""
		src = options.get("src")

		if src is None or self.placement[src] == self.rank:
			return

		# If some metadata is missing, it should be received from the previous block before actual data
		if any(not var.metadata for var in self.input_variables):
			self._receive_metadata(src)

		recvs = []
		# We couple buffer and work objects for now so that we can wait at the right moment
		for var in self.input_variables:
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

			var.set(var.to_process, mb_id, (work, buffer))

		return recvs  # if not batched, recvs is still empty and therefore Falsy

	def recv_backward(self, mb_id, mb_size, **options):
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
		for var in self.output_variables:
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

				target.set(target.to_process, mb_id, (work, buffer))

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
		Register input metadata by receiving it from the previous block. No-op if src is None or the same as the current rank.

		:param src: rank of the previous block
		:type src: int
		"""

		for var in self.input_variables:
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

		logger.debug(f"{self} - Registered metadata {[var for var in self.input_variables]} from {src}")

	def _send_metadata(self, dst):
		"""
		Send output metadata to the next block. No-op if dst is None or the same as the current rank.

		:param dst: rank of the next block
		:type dst: int
		"""
		if dst is None or self.placement[dst] == self.rank:
			return

		for var in self.output_variables:
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
		for var, value in zip(self.output_variables, output):
			for target in var:
				target.metadata = TensorMetadata(
					value[0]
				)  # omit batch dimension ; if the tensor is not batched, this is wrong!
			logger.debug(f"{self} - registered output metadata {var[0].metadata} for {var[0].name}")
