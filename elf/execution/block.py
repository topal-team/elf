"""
Individual stage computation and communication management
"""

import torch
import torch.distributed as dist

from elf.scheduling import OpOptions
from elf.utils import PreHide, Timer, TensorMetadata, _is_mpi, Hide
from elf.zb_utils import LayerDW, partial_dx_recomputation, recording
from elf.execution.remat import Stage, unit_placement

import logging

logger = logging.getLogger(__name__)


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
			raise Exception(
				f"Variable {self.name} - Trying to pop data at mb_id {mb_id}, but it's empty ({self._state()})"
			)

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

	def __init__(self, model, id_, signature, topology):
		"""
		:param model: layer / group of layers that will perform the computation
		:type model: nn.Module
		:param id_: number of this block in the pipeline
		:type id_: int
		:param signature: signature of the block
		:type signature: Signature
		:param topology: process topology of the pipeline this block belongs to. The block reads its
			placement and all the process groups it communicates on from it.
		:type topology: PipelineTopology
		"""
		super(PipelineBlock, self).__init__()
		# Block infos
		self.topology = topology
		self.placement = topology.placement  # need it to resolve src/dst later on
		self.rank = self.placement[id_]  # global rank
		self.model = model.cuda() if torch.cuda.is_available() else model

		self.id = id_  # rank in the model
		self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
		self.is_last = self.id == len(self.placement) - 1
		self.is_first = self.id == 0

		# Process groups for collective communications
		self.pp_group = topology.pp_group
		self.dp_group = topology.dp_group(id_)

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

		# Process groups for point-to-point communications, one per peer and direction
		self.recv_pgs, self.send_pgs = topology.p2p_groups(id_)

		self.compute_time = []  # used to measure idle time

		# Take care of rematerialization
		self.stages = []  # one per micro-batch

		self.send_ops = []  # need to be stored for MPI backend, see _wait_for_send_ops

		self.all_reduce_work = None  # all reduce is enqueued asynchronously to make sure it runs on a separate stream, but needs to be synced at the end of the iteration

		# NCCL automatically synchronizes send stream with compute stream
		# With MPI however, we need to manually make sure that computations are finished before sending. These events are used for that.
		# note: we could optimize that a bit by:
		# 1 - having a different stream for fwd and bwd (to avoid waiting for stuff we don't depend on)
		# 2 - finding a way to insert events just after the computation that leads to the sendable tensor, instead of waiting for the whole computation to be finished (not sure if possible)
		self.fwd_compute_events = []
		self.bwd_compute_events = []

		# Find and cache all LayerDWs for ZB control
		self._layers_dw = set()
		for module in self.model.modules():
			if isinstance(module, LayerDW):
				self._layers_dw.add(module)

	def __str__(self) -> str:
		return f"Rank {self.rank} - Layer {self.id}"

	def forward(self, mb_id, **options):
		"""
		Forward pass through this stage

		Possible options:
		- REMAT_STRATEGY: function(name, module) -> RematMode | None - How each module's activations are handled, None leaving the module alone
		"""
		# Gather all variables needed for forward
		inputs = []
		for input_var in self.input_variables:
			logger.debug(f"{self} - Waiting for {input_var}")
			x = input_var.wait_and_pop(mb_id)

			if x.is_floating_point() or x.is_complex():  # int tensors can't require grads
				x.requires_grad_(True)

			inputs.append(PreHide(x))

		# Place the rematerialization units. A forward whose activations something rebuilds later
		# keeps what that replay starts from, so the block does not have to hold its own inputs.
		replayable = options.get(OpOptions.REPLAYABLE, False)
		strategy = unit_placement(options.get(OpOptions.REMAT_STRATEGY), replayable=replayable)
		stage = Stage(self.model, strategy, replayable=replayable)
		if len(self.stages) < mb_id + 1:
			self.stages.extend([None] * (mb_id - len(self.stages) + 1))
		self.stages[mb_id] = stage

		y = self._compute_forward(stage, inputs, f"forward({self.id}:{mb_id})")

		if torch.cuda.is_available() and _is_mpi():
			self.fwd_compute_events.append(torch.cuda.current_stream().record_event())

		if any(not dst.metadata for var in self.output_variables for dst in var):
			self._register_out_metadata(y)

		# Register outputs to send
		for output_var, value in zip(self.output_variables, y):
			for output_dst in output_var:
				output_dst.set(output_dst.to_send, mb_id, value.detach())

		# Always save inputs
		# Which are PreHide objects, so it does not cost memory
		for input_var, value in zip(self.input_variables, inputs):
			input_var.set(input_var.saved, mb_id, value)

		hidden = Hide(y)  # single node for tuple of outputs
		# save it in the first output variables, even if it contains all outputs
		var = self.output_variables[0][0]
		var.set(var.saved, mb_id, hidden)  # save only to first, we only need one

		# If this is the last block, return the output
		if self.is_last:
			output = []
			for output_var in self.output_variables:
				assert len(output_var) == 1, "Output of the pipeline has multiple destinations"
				output.append(output_var[0].get(output_var[0].to_send, mb_id))
			return output

	def _compute_forward(self, stage, inputs, timer_name="forward"):
		"""
		Return the outputs of the forward pass of the model as a tuple

		:param stage: rematerialization units to run the model under
		:type stage: elf.execution.remat.Stage
		:param inputs: inputs to the model
		:type inputs: Tuple[Tensor, Any]
		:return: output of the forward pass
		:rtype: Tuple[Tensor, Any]
		"""
		with Timer(
			name=f"{timer_name}"
		) as timer:  # can we avoid event creation when ELF_TIMINGS is not set?
			inputs = tuple(x.get() for x in inputs)  # inputs are PreHide objects
			y = stage.run(*inputs)
			y = (y,) if not isinstance(y, tuple) else y

		self.compute_time.append(timer)
		return y

	def _compute_backward_inputs(
		self, inputs, outputs, grads, timer_name="backward_inputs", retain_graph=False
	):
		"""
		Return the grads of the inputs of the model as a tuple

		:param retain_graph: keep the graph alive, for a schedule that walks it again before W
		:type retain_graph: bool
		"""
		# assert len(outputs) == len(grads), (
		# 	"Outputs and grads must have the same length (got %d, %d)" % (len(outputs), len(grads))
		# )

		with Timer(name=f"{timer_name}") as timer:
			outputs.backward(grads, retain_graph=retain_graph)

		grads = []
		for input in inputs:
			if input.requires_grad:
				grads.append(input.grad.data)

		self.compute_time.append(timer)
		return grads

	def recompute_forward(self, mb_id, **options):
		"""
		Rebuild the activations this stage is not holding, by replaying its forward.
		This can be scheduled both between F and B, or between B and W. In any case, the graph is kept, and values are simply refilled.
		"""
		stage = self.stages[mb_id]
		filling = [module for module in self._layers_dw if module.needs_inputs(mb_id)]
		missing = [unit for unit in stage.units if unit.needs_filling()]

		with Timer(name=f"recompute_forward({self.id}:{mb_id})") as timer:
			if (
				not filling and all(unit.can_refill() for unit in missing)
			):  # every unit can already recompute on its own ; no need to do a full stage replay, as it might start additional computations
				for unit in missing:
					if unit.needs_filling():  # may have been filled by a previous one cascading
						unit.recompute()
			else:
				for module in filling:
					module.begin_filling(mb_id)
				try:
					stage.replay()
				finally:
					for module in filling:
						module.end_filling()

		self.compute_time.append(timer)

	def recompute_backward_inputs(self, mb_id, **options):
		"""
		Recompute the backward pass for the inputs of the model
		"""
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

				# Partial DX recomputation recomputes the minimal backward pass needed to compute W, instead of the full backward pass
				# We can't use torch.grad with inputs=inputs, because if the inputs don't require grads (e.g., int tensors), then torch.grad will fail
				partial_dx_recomputation(output, grad, retain_graph=retain_graph)

		self.compute_time.append(timer)

		for module in self._layers_dw:
			module.commit(mb_id)

	def backward_params(self, mb_id, **options):
		"""
		Perform the backward pass for the parameters of the model

		:param mb_id: micro-batch index
		:type mb_id: int
		:param **options: options to modify the backward behaviour
		"""
		with Timer(name=f"backward_params({self.id}:{mb_id})") as timer:
			for module in self._layers_dw:
				if not module.is_empty(mb_id):  # skip modules that are not involved in the computation
					module.backward(mb_id)

		self.compute_time.append(timer)

		# nothing else will be asked of this micro-batch
		if mb_id < len(self.stages):
			self.stages[mb_id] = None

	def send_forward(self, mb_id, dst, **options):
		"""
		Send one activation to the next layer in the model

		:param mb_id: micro-batch index to send
		:type mb_id: int
		:param dst: block this activation is sent to, or None when it leaves the pipeline
		:type dst: int or None
		:param **options: options to modify the send behaviour
		"""
		if torch.cuda.is_available() and _is_mpi():
			# This is a sync point that is needed for MPI but not for NCCL
			self.fwd_compute_events[mb_id].synchronize()  # wait for computation to be finished

		if dst is None or self.placement[dst] == self.rank:
			return

		for var in self.output_variables:
			for target in var:
				# only perform communications for that dst
				if target.peer != dst:
					continue

				if not target.was_metadata_sent:
					self._send_metadata(dst)
				outputs = target.get(target.to_send, mb_id).contiguous()

				rank = self.placement[dst]  # we now use the actual rank instead of the block id
				logger.debug(f"{self} - Sending outputs to rank {rank}")
				work = dist.isend(outputs, rank, group=self.send_pgs["fwd"][dst])
				if _is_mpi():
					self.send_ops.append(work)

	def send_backward(self, mb_id, dst, **options):
		"""
		Send one gradient tensor to the previous layer in the model

		:param mb_id: micro-batch index to send
		:type mb_id: int
		:param dst: block this gradient is sent to, or None when it leaves the pipeline
		:type dst: int or None
		:param **options: options to modify the send behaviour
		"""
		if torch.cuda.is_available() and _is_mpi():
			self.bwd_compute_events[mb_id].synchronize()  # wait for computation to be finished

		if dst is None or self.placement[dst] == self.rank:
			return

		for var in self.input_variables:
			if var.peer != dst:
				continue

			grads = var.get(var.to_send, mb_id).contiguous()

			rank = self.placement[dst]
			logger.debug(f"{self} - Sending gradients to rank {rank}")
			work = dist.isend(grads, rank, group=self.send_pgs["bwd"][dst])
			if _is_mpi():
				self.send_ops.append(work)

	def recv_forward(self, mb_id, mb_size, src, **options):
		"""
		Receive and store one activation to forward

		:param mb_id: micro-batch index to receive
		:type mb_id: int
		:param mb_size: size of the micro batch to receive
		:type mb_size: int
		:param src: block this activation comes from, or None when it enters the pipeline
		:type src: int or None
		:param **options: options to modify the receive behaviour
		"""
		if src is None or self.placement[src] == self.rank:
			return

		# If some metadata is missing, it should be received from the previous block before actual data
		if any(not var.metadata and var.peer == src for var in self.input_variables):
			self._receive_metadata(src)

		# We couple buffer and work objects for now so that we can wait at the right moment
		for var in self.input_variables:
			if var.peer != src:
				continue

			buffer = var.metadata.get_buffer(mb_size)

			rank = self.placement[src]
			logger.debug(f"{self} - Starting to receive inputs from rank {rank}")
			work = dist.irecv(buffer, rank, group=self.recv_pgs["fwd"][src])

			var.set(var.to_process, mb_id, (work, buffer.requires_grad_(True)))

	def recv_backward(self, mb_id, mb_size, src, **options):
		"""
		Receive and store one gradient to backward

		:param mb_id: micro-batch index to receive
		:type mb_id: int
		:param mb_size: size of the micro batch to receive
		:type mb_size: int
		:param src: block this gradient comes from, or None when it leaves the pipeline
		:type src: int or None
		:param **options: options to modify the receive behaviour
		"""
		if src is None or self.placement[src] == self.rank:
			return

		for var in self.output_variables:
			for target in var:
				if target.peer != src:
					continue

				buffer = target.metadata.get_buffer(mb_size)

				rank = self.placement[src]
				logger.debug(f"{self} - Starting to receive gradients from rank {rank}")
				work = dist.irecv(buffer, rank, group=self.recv_pgs["bwd"][src])

				target.set(target.to_process, mb_id, (work, buffer))

	def all_reduce_param_grads(self, **options):
		"""
		All-reduce operation on the gradients of the model parameters across the data parallel group.
		:param **options: Additional options to modify the all-reduce behaviour.
		:type **options: dict
		"""
		if self.dp_group is None:
			return
		grads = [p.grad for _, p in sorted(self.model.named_parameters()) if p.grad is not None]
		self.all_reduce_work = dist.all_reduce_coalesced(
			grads, op=dist.ReduceOp.AVG, group=self.dp_group, async_op=True
		)

	def sync_all_reduce(self):
		if self.all_reduce_work is None:
			return
		self.all_reduce_work.wait()
		self.all_reduce_work = None

	def scale_grads(self, batch_size):
		"""
		Scale the gradients of the model parameters by the batch size

		:param batch_size: size of the full batch
		:type batch_size: int
		"""
		grads = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
		if not grads:
			return
		torch._foreach_div_(grads, batch_size)  # torch.foreach.div_ only exists in torch >= 2.15

	def _receive_metadata(self, src):
		"""
		Register input metadata by receiving it from the previous block. No-op if src is None or the same as the current rank.

		:param src: rank of the previous block
		:type src: int
		"""
		if src is None or self.placement[src] == self.rank:
			return

		for var in self.input_variables:
			if var.peer != src:
				continue

			rank = self.placement[src]  # we resolve the actual rank now
			logger.debug(f"{self} - Receiving metadata from {rank}")
			metadata = torch.empty(TensorMetadata.MAX_SIZE, device=self.device)
			dist.recv(metadata, src=rank, group=self.recv_pgs["fwd"][src])
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
				dist.send(target.metadata.to_tensor(), dst=rank, group=self.send_pgs["fwd"][dst])
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

	def _wait_for_send_ops(self):
		"""
		With MPI process groups, tensors sent by a P2P are held by the AsyncWork object.
		If we don't keep a reference to them, they are garbage collected and the memory is lost.
		Instead, we keep a reference to the AsyncWork object and wait for it at the end (when it won't slow down anything).
		"""
		for op in self.send_ops:
			op.wait()
		self.send_ops.clear()
		self.fwd_compute_events.clear()
		self.bwd_compute_events.clear()

	def backward_inputs(self, mb_id, **options):
		"""
		Perform the backward pass for the inputs of the model

		Possible options:
		- RECOMPUTE_ACTIVATIONS: bool - Whether to recompute the activations of the model before W
		- RECOMPUTE_GRADIENTS: bool - Whether to recompute the gradients of the activations before W
		"""
		# Get strategies from options
		recompute_activations = options.get(OpOptions.RECOMPUTE_ACTIVATIONS, False)
		recompute_gradients = options.get(OpOptions.RECOMPUTE_GRADIENTS, False)
		assert recompute_activations or not recompute_gradients, (
			"recomputing the gradients before W means walking the graph again, which needs the "
			"activations they come from: RECOMPUTE_GRADIENTS implies RECOMPUTE_ACTIVATIONS"
		)

		# We need inputs to get their gradients later on
		inputs = []
		for input_var in self.input_variables:
			value = input_var.get(input_var.saved, mb_id)
			inputs.append(value)

		# We need outputs to start the backward
		# it should be a Hide object stored in the first variable+target
		var = self.output_variables[0][0]
		# RB starts from the same node, so the graph entry point has to survive this backward --
		# but only when there is going to be one. Nothing else ever reads it, and holding it keeps
		# the nodes of a graph whose activations this backward has already consumed.
		outputs = var.get(var.saved, mb_id, delete=not recompute_gradients)

		# Gather the incoming gradients
		# If an output has multiple destinations, we sum up the gradients
		# Maybe we should time this and add to compute time, but it's probably negligible
		output_grads = []
		for output_var in self.output_variables:
			logger.debug(f"{self} - Waiting for {output_var[0]}")
			grad_accumulator = output_var[0].wait_and_pop(mb_id)
			for dst in output_var[1:]:
				logger.debug(f"{self} - Waiting for {dst}")
				grad = dst.wait_and_pop(mb_id)
				grad_accumulator += grad
			output_grads.append(grad_accumulator)

		# keep only what nothing else is going to rebuild: an RB supersedes every gradient this
		# backward could record, and an RF gives the inputs back. Leaving the inputs out is what
		# keeps this backward from holding the whole activation set until it ends. The graph is
		# kept for that RB, and for nothing else.
		with recording(gradients=not recompute_gradients, inputs=not recompute_activations):
			input_grads = self._compute_backward_inputs(
				inputs,
				outputs,
				output_grads,
				f"backward_inputs({self.id}:{mb_id})",
				retain_graph=recompute_gradients,
			)

		if torch.cuda.is_available() and _is_mpi():
			self.bwd_compute_events.append(torch.cuda.current_stream().record_event())

		for module in self._layers_dw:
			module.commit(mb_id)

		# Register the gradients to send
		for input_var, value in zip(self.input_variables, input_grads):
			input_var.set(input_var.to_send, mb_id, value)

		# The activations are rebuilt by replaying the forward, so they can go now. The stage keeps
		# what that replay needs; without a replay ahead of it, it keeps nothing.
		if recompute_activations:
			self.stages[mb_id].drop()
		else:
			self.stages[mb_id].forget()

		if recompute_gradients:
			for output_var, value in zip(self.output_variables, output_grads):
				output_var[0].set(output_var[0].to_process, mb_id, value)
