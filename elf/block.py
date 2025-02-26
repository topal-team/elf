"""
Individual stage computation and communication management
"""

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from contextlib import contextmanager

from .scheduling import OpOptions
from .utils import Timer, TensorMetadata, recompute_all_context_fn
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

	def get(self, queue, mb_id, delete=True):
		"""
		Get an element from the specified queue at the given micro-batch index.

		:param queue: Queue to get data from (to_process, saved, or to_send)
		:type queue: List
		:param mb_id: Micro-batch index to retrieve
		:type mb_id: int
		:param delete: Whether to delete the data from the queue
		:type delete: bool
		:return: Data at the specified index
		:rtype: torch.Tensor or Any
		:raises Exception: If queue is shorter than mb_id or if data at mb_id is None
		"""
		if len(queue) <= mb_id:
			raise Exception(f"Variable {self.name} - Data queue shorter than mb_id {mb_id}")
		if queue[mb_id] is None:
			raise Exception(f"Variable {self.name} - Trying to pop data at mb_id {mb_id}, but it's empty")

		data = queue[mb_id]
		if delete:
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
		Basic forward pass operation
		"""
		# Gather all variables needed for forward
		inputs = []
		for input_var in self.input_variables:
			x = input_var.wait_and_pop(mb_id)

			if x.is_floating_point():
				x.requires_grad_(True)

			inputs.append(x)

		# By default, checkpoint nothing. Otherwise, use the provided strategy
		remat_strategy = options.get(OpOptions.REMAT_STRATEGY, lambda _: False)

		with self._selective_remat(remat_strategy, mb_id):
			y = self._compute_forward(inputs, f"forward({self.id}:{mb_id})")

		if any(not dst.metadata for var in self.output_variables for dst in var):
			self._register_out_metadata(y)

		for module in self.model.modules():
			if isinstance(module, LayerDW):
				module.move_last_computed("input", mb_id)

		# Register outputs to send
		for output_var, value in zip(self.output_variables, y):
			for output_dst in output_var:
				output_dst.set(output_dst.to_send, mb_id, value.detach())

		# Always save inputs
		for input_var, value in zip(self.input_variables, inputs):
			input_var.set(input_var.saved, mb_id, value)

		# Save outputs for backward, except if specified otherwise
		save = options.get(OpOptions.SAVE, True)
		if save:
			for output_var, value in zip(self.output_variables, y):
				output_var[0].set(output_var[0].saved, mb_id, value)  # save only to first, we only need one
		else:
			for module in self.model.modules():
				if isinstance(module, LayerDW):
					module.delete("input", mb_id)

		# If this is the last block, return the output
		if self.is_last:
			output = []
			for output_var in self.output_variables:
				assert len(output_var) == 1, "Output of the pipeline has multiple destinations"
				output.append(output_var[0].get(output_var[0].to_send, mb_id))
			return output

	@contextmanager
	def _selective_remat(self, remat_strategy, mb_id):
		# Save original forwards and wrap with checkpoint
		for module in self.model.modules():
			if remat_strategy(module):
				original = getattr(module, "forward")
				setattr(module, "_elf_original_forward", original)

				def wrapped_forward(*args, **kwargs):
					return checkpoint(
						original,
						*args,
						**kwargs,
						use_reentrant=False,
						context_fn=recompute_all_context_fn(),
					)
					

				setattr(module, "forward", wrapped_forward)
		try:
			yield
		finally:
			for module in self.model.modules():
				if remat_strategy(module):
					# Delete unwanted activations
					# (Should we iterate over submodules here?)
					if isinstance(module, LayerDW):
						module.delete("input", mb_id)
					# Restore original forwards
					setattr(module, "forward", getattr(module, "_elf_original_forward"))
					delattr(module, "_elf_original_forward")

	def backward_inputs(self, mb_id, **options):
		"""
		Perform the backward pass for the inputs of the model
		"""
		# We need inputs to get their gradients later on
		# TODO: we don't actually need their data! find a way to get gradients without them (maybe GradientEdge?)
		inputs = []
		for input_var in self.input_variables:
			value = input_var.get(input_var.saved, mb_id)
			if value.requires_grad:
				inputs.append(value)

		# We need outputs to start the backward
		outputs = []
		for output_var in self.output_variables:
			value = output_var[0].get(output_var[0].saved, mb_id)
			outputs.append(value)

		# Gather the incoming gradients
		# If an output has multiple destinations, we sum up the gradients
		# Maybe we should time this and add to compute time, but it's probably negligible
		output_grads = []
		for output_var in self.output_variables:
			grad_accumulator = output_var[0].wait_and_pop(mb_id)
			for dst in output_var[1:]:
				grad = dst.wait_and_pop(mb_id)
				grad_accumulator += grad
			output_grads.append(grad_accumulator)

		input_grads = self._compute_backward_inputs(
			inputs, outputs, output_grads, f"backward_inputs({self.id}:{mb_id})"
		)

		for module in self.model.modules():
			if isinstance(module, LayerDW):
				module.move_last_computed("grad_output", mb_id)

		# Register the gradients to send
		for input_var, value in zip(self.input_variables, input_grads):
			input_var.set(input_var.to_send, mb_id, value)

		save = options.get(OpOptions.SAVE, "full")
		# can be "full", "gradients" or "none"

		# If we don't save activations, we need to keep the inputs for recomputation
		if save != "full":
			for input_var, value in zip(self.input_variables, inputs):
				input_var.set(input_var.saved, mb_id, value)
		
		if save == "none":
			# save incoming grads for second computation
			for output_var, grad in zip(self.output_variables, output_grads):
				output_var[0].set(output_var[0].to_process, mb_id, grad)

		for module in self.model.modules():
			if isinstance(module, LayerDW):
				if save != "full":  # will recompute F
					module.delete("input", mb_id)
				if save == "none":  # will recompute F+B
					module.delete("grad_output", mb_id)

	def _compute_forward(self, inputs, timer_name="forward"):
		"""
		Return the outputs of the forward pass of the model as a tuple

		:param inputs: inputs to the model
		:type inputs: Tuple[Tensor, Any]
		:return: output of the forward pass
		:rtype: Tuple[Tensor, Any]
		"""
		with Timer(name=f"{timer_name}") as timer:
			y = self.model(*inputs)
			y = (y,) if not isinstance(y, tuple) else y

		self.compute_time.append(timer)
		return y

	def _compute_backward_inputs(self, inputs, outputs, grads, timer_name="backward_inputs"):
		"""
		Return the grads of the inputs of the model as a tuple
		"""
		assert len(outputs) == len(grads), (
			"Outputs and grads must have the same length (got %d, %d)" % (len(outputs), len(grads))
		)

		with Timer(name=f"{timer_name}") as timer:
			for i, (output, grad) in enumerate(zip(outputs, grads)):
				assert output.shape == grad.shape, (
					f"Output and grad shapes do not match: {output.shape} != {grad.shape}"
				)
				retain_graph = i != len(outputs) - 1  # save until the last backward
				torch.autograd.backward(output, grad, retain_graph=retain_graph)

		grads = []
		for input in inputs:
			if input.requires_grad:
				grads.append(input.grad.data)
			else:
				logger.warning(f"{self} - No gradient computed for var {input}")

		self.compute_time.append(timer)
		return grads

	def recompute_forward(self, mb_id, **options):
		"""
		Recompute the forward pass of the model
		"""
		# Inputs should have been saved by a previous forward pass
		inputs = []
		for input_var in self.input_variables:
			# Keep it for backward
			value = input_var.get(input_var.saved, mb_id)
			inputs.append(value)

		# if we're doing B next, we need to save inputs/outputs
		# if we're doing W without B however, we don't need to save inputs/outputs
		save = options.get(OpOptions.SAVE, False)
		if not save:
			with torch.no_grad():
				y = self._compute_forward(inputs, f"recompute_forward({self.id}:{mb_id})")
		else:
			y = self._compute_forward(inputs, f"recompute_forward({self.id}:{mb_id})")

		for module in self.model.modules():
			if isinstance(module, LayerDW):
				module.move_last_computed("input", mb_id)

		if save:
			# Save inputs and outputs for second backward
			for input_var, value in zip(self.input_variables, inputs):
				input_var.set(input_var.saved, mb_id, value)
				
			for output_var, value in zip(self.output_variables, y):
				for output_dst in output_var:
					output_dst.set(output_dst.saved, mb_id, value)


	def recompute_backward_inputs(self, mb_id, **options):
		"""
		Recompute the backward pass for the inputs of the model
		"""
		inputs = []
		for input_var in self.input_variables:
			value = input_var.get(input_var.saved, mb_id)
			inputs.append(value)

		outputs = []
		for output_var in self.output_variables:
			value = output_var[0].get(output_var[0].saved, mb_id)
			outputs.append(value)

		grads = []
		for output_var in self.output_variables:
			# no need to accumulate here, it was already done in the first backward
			grad = output_var[0].get(output_var[0].to_process, mb_id)
			grads.append(grad)

		# Don't use _compute_backward_inputs here because we don't want to compute grads w.r.t parameters that are not LayerDWs
		with Timer(name=f"recompute_backward_inputs({self.id}:{mb_id})") as timer:
			for i, (output, grad) in enumerate(zip(outputs, grads)):
				retain_graph = i != len(outputs) - 1  # save until the last backward
				torch.autograd.backward(output, grad, inputs=inputs, retain_graph=retain_graph)
				
		self.compute_time.append(timer)

		for module in self.model.modules():
			if isinstance(module, LayerDW):
				module.move_last_computed("grad_output", mb_id)

	def backward_params(self, mb_id, **options):
		"""
		Perform the backward pass for the parameters of the model

		:param mb_id: micro-batch index
		:type mb_id: int
		:param **options: options to modify the backward behaviour
		"""
		with Timer(name=f"backward_params({self.id}:{mb_id})") as timer:
			for module in self.model.modules():
				if isinstance(module, LayerDW):
					module.backward(mb_id)

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

			var.set(var.to_process, mb_id, (work, buffer.requires_grad_(True)))

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

	def _offload_dw(self, to="cpu"):
		"""
		Offload the gradients and activations saved between B and W. The computation will be done on the new device.

		:param to: device to offload to
		:type to: str or torch.device
		"""
		for module in self.model.modules():
			if isinstance(module, LayerDW):
				module.offload_last(to)
