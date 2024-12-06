"""
Execution manager
"""

import torch
import torch.distributed as dist

from collections import deque
import time

from .scheduling import OperationType, OpOptions
from .utils import Timer, op_to_str

import logging

logger = logging.getLogger("engine")


def _fake_p2p(data):
	"""
	Simulates P2P communication by creating a fake work.

	:param data: Input data to be processed
	:type data: Iterator[Tuple[str, Tensor]]
	:return: Tuple with fake communication buffers
	:rtype: Tuple[Tensor]
	"""
	return [None, data.detach()]


class Engine:
	"""
	Coordinates the execution of a schedule on a list of blocks at a device/rank level.
	Takes care of feeding the input to the first block and computing the loss on the last block.
	"""

	def __init__(self, blocks):
		"""
		:param blocks: list of blocks handled by this process rank
		:type blocks: List[PipelineBlock]
		"""
		self.blocks = blocks
		self.rank = self.blocks[0].rank if blocks else None
		for b in self.blocks:
			assert b.rank == self.rank, "All blocks in a stage should be on the same rank"
		self.id_to_block = {b.id: b for b in self.blocks}
		self.comms = {}

	def _add_comm(self, comm, id_):
		if not comm:
			return

		if id_ not in self.comms:
			self.comms[id_] = []
		self.comms[id_].extend(comm)

	def _run_comms(self):
		"""
		Run all currently batched communications for this device
		Internal function, this should not be used by the user
		"""
		if len(self.comms) == 0:
			return
		
		streams = []

		# Enqueue everything
		for id_, comms in self.comms.items():
			# We only run batched communications if we have both send and receive
			if any(c.op.__name__ == "isend" for c in comms) and any(
				c.op.__name__ == "irecv" for c in comms
			):
				stream = torch.cuda.Stream()
				streams.append(stream)
				with torch.cuda.stream(stream):
					if len(comms) > 1:
						logger.debug(
							f"Rank {self.rank} - Running batched communications with id {id_}: {[op_to_str(c) for c in comms]}"
						)
					works = dist.batch_isend_irecv(comms)
					for w in works:
						w.wait()
						
				self.comms[id_] = []

		# Synchronize everything
		for stream in streams:
			stream.synchronize()

		logger.debug(f"Rank {self.rank} - Finished batched communications")

		# Clean up
		keys = list(self.comms.keys())
		for k in keys:
			if len(self.comms[k]) == 0:
				del self.comms[k]

	def train_step(self, batch, target, loss_fn, schedule, mb_sizes, profile=False):
		"""
		Executes a schedule on a batch of data

		:param batch: input data, only used on the first block of the pipeline
		:type batch: List[Tensor]
		:param target: groundtruth, only used on the last block of the pipeline
		:type target: List[Tensor]
		:param loss_fn: loss function to use ; we recommend using the torch built-in function, but if you want to use your own make sure that summing the loss of every micro-batch independently is equivalent to the loss on the entire batch (e.g. no average across batches). The loss is averaged across the entire batch at the end.
		:type loss_fn: function (Tensor, Tensor) -> Tensor
		:param schedule: list of operations. For more info, see schedule.
		:type schedule: list[Operation]
		:param mb_sizes: list of micro batch sizes. The list should cover the entire batch, i.e. ``sum(mb_sizes) == batch_size``
		:type mb_sizes: int or List[int]
		:param profile: Whether to activate nvidia profiling or not. If True, NVTX ranges will be generated for each operation
		:type profile: boolean

		:return:

		- Result of the forward pass
		- Losses for each micro-batch
		- Insights about time taken, as a dict containing:


			- total: total time taken for the execution
			- idle: total time not used for computation for this process
			- start_idle: time between the start of execution and the first computation
			- end_idle: time between the last computation and the end of execution
			- bubble: idle time between first and last computation

		:rtype: Tensor, Tensor, Dict[float]
		"""
		split_batches = [tensor.split(mb_sizes, dim=0) for tensor in batch]
		microbatches = iter(zip(*split_batches))
		microtargets = target.split(mb_sizes, dim=0)

		result = []
		losses = []

		grad_fns = deque()

		dist.barrier()  # useful for timing, but it probably slows down the execution a bit
		pipe_start = time.time()
		warmup_start = None

		for op in schedule:
			block = self.id_to_block.get(op.block_id)
			if block is None:
				continue  # not my job

			logger.debug(f"Computing {op} on block {block}")

			if profile:
				torch.cuda.nvtx.range_push(f"{block}:{op}")
			if warmup_start is None and op.op != OperationType.RECV_FORWARD:
				# Warmup time is the time spent waiting for the first forward
				# The first operation after that is the end of warmup
				torch.cuda.synchronize()
				warmup_start = time.time()

			# A computation will synchronize! We want to enqueue all possible communications before that
			if op.op in [OperationType.FORWARD, OperationType.BACKWARD]:
				self._run_comms()

			match op.op:
				case OperationType.FORWARD:
					y = block.forward(op.mb_id, **op.options)
					# If the block as multiple outputs, this flattens them
					# TODO: correctly handle that in multiple result lists
					if y is not None:
						for output in y:
							if isinstance(output, torch.Tensor):
								result.append(output.detach().requires_grad_(False))
							else:
								logger.warning("Non-tensor output")
								result.append(output)

				case OperationType.BACKWARD:
					block.backward(op.mb_id, **op.options)

				case OperationType.SEND_FORWARD:
					if op.options.get("dst") in self.id_to_block:
						# The destination block is on the same device ; we bypass p2p comms
						dst_block = self.id_to_block.get(op.options.get("dst"))
						_transfer_forward(block, dst_block, op.mb_id)

					comm = block.send_forward(op.mb_id, **op.options)
					self._add_comm(comm, op.options.get(OpOptions.BATCHED_COMM))

				case OperationType.SEND_BACKWARD:
					if op.options.get("dst") in self.id_to_block:
						# The destination block is on the same device ; we bypass p2p comms
						dst_block = self.id_to_block.get(op.options.get("dst"))
						_transfer_backward(block, dst_block, op.mb_id)

					comm = block.send_backward(op.mb_id, **op.options)
					self._add_comm(comm, op.options.get(OpOptions.BATCHED_COMM))

				case OperationType.RECV_FORWARD:
					if block.is_first:
						microbatch = next(microbatches)
						for mb, var in zip(microbatch, block.inputs):
							var.set(var.to_process, op.mb_id, _fake_p2p(mb))

					comm = block.recv_forward(op.mb_id, mb_sizes[op.mb_id], **op.options)
					self._add_comm(comm, op.options.get(OpOptions.BATCHED_COMM))

				case OperationType.RECV_BACKWARD:
					comm = block.recv_backward(op.mb_id, mb_sizes[op.mb_id], **op.options)
					self._add_comm(comm, op.options.get(OpOptions.BATCHED_COMM))

				case OperationType.LOSS_FORWARD:
					if block.is_last:
						assert op.mb_id < len(
							result
						), f"Loss forward for mb {op.mb_id} but only {len(result)} results computed"
						loss, grad_fn = compute_loss(block, result[op.mb_id], microtargets[op.mb_id], loss_fn)
						losses.append(loss)
						grad_fns.append(grad_fn)
						logger.debug(f"{block} - Computed loss = {loss.item()}")
					else:
						logger.warning(f"Tried to compute loss on a non-last block {block}")
						continue

				case OperationType.LOSS_BACKWARD:
					if block.is_last:
						assert op.mb_id < len(
							losses
						), f"Loss backward for mb {op.mb_id} but only {len(losses)} losses computed"
						grad_fn = grad_fns.popleft()
						with Timer() as timer:
							grads = grad_fn()
						block.compute_time.append(timer.time)
						for grad, var in zip(grads, block.outputs):
							for dst in var:  # should be only one destination
								dst.set(dst.to_process, op.mb_id, _fake_p2p(grad))
					else:
						logger.warning(f"Tried to compute loss backward on a non-last block {block}")
						continue

				case OperationType.ALL_REDUCE_PARAM_GRADS:
					block.scale_grads(sum(mb_sizes))  # we also average out the gradients here
					block.all_reduce_param_grads(**op.options)

				case _:
					raise Exception(f"Unknown operation : {op}")

			if profile:
				torch.cuda.nvtx.range_pop()

		logger.debug(f"[Rank {self.rank}] - Finished execution")

		self._run_comms()  # finish all comms
		torch.cuda.synchronize()
		cooldown_end = time.time()
		dist.barrier()
		pipe_end = time.time()

		compute_time = 0
		for block in self.blocks:
			compute_time += sum([f() for f in block.compute_time])
			block.compute_time = []

		times = {
			"total": pipe_end - pipe_start,
			"idle": pipe_end - pipe_start - compute_time,
			"start_idle": warmup_start - pipe_start,
			"end_idle": pipe_end - cooldown_end,
		}
		times["bubble"] = times["idle"] - times["start_idle"] - times["end_idle"]
		return result, losses, times


def compute_loss(block, output, target, loss_fn):
	"""
	Computes the loss value and prepares a function to compute the gradients with respect to the block's outputs.

	:param block: last block of the pipeline
	:type block: PipelineBlock
	:param output: output of this block
	:type output: Tensor
	:param target: target value
	:type target: Tensor
	:param loss_fn: loss function to compute
	:type loss_fn: function (Tensor, Tensor, reduction = 'sum') -> Tensor

	:return: loss value, gradient function
	:rtype: Tensor, Callable[[], Tuple[Tensor]]
	"""
	if not isinstance(output, torch.Tensor) or not isinstance(target, torch.Tensor):
		logger.warning("Loss computation with non-tensor output or target")

	output = output.detach().requires_grad_()

	with Timer() as timer:
		try:
			loss = loss_fn(output, target, reduction="sum")
		except TypeError:
			loss = loss_fn(output, target)

	block.compute_time.append(timer.time)
	loss = loss / (target.numel() // target.size(0))

	def grad_fn():
		loss.backward()
		return (output.grad.data,)

	return loss.detach().requires_grad_(False), grad_fn


def _transfer_forward(block, dst_block, mb_id):
	"""
	P2P communications bypass for same-rank blocks that need to communicate

	:param block: source block
	:type block: PipelineBlock
	:param dst_block: target block
	:type dst_block: PipelineBlock
	"""
	# We match by order and not by name because the names can be different with pre-partitioned models
	all_to_send = [dst for var in block.outputs for dst in var if dst.peer == dst_block.id]
	all_to_recv = [var for var in dst_block.inputs if var.peer == block.id]
	for src, dst in zip(all_to_send, all_to_recv):
		inputs = src.get(src.to_send, mb_id)
		dst.set(dst.to_process, mb_id, _fake_p2p(inputs))


def _transfer_backward(block, dst_block, mb_id):
	"""
	P2P communications bypass for same-rank blocks that need to communicate

	:param block: source block
	:type block: PipelineBlock
	:param dst_block: target block
	:type dst_block: PipelineBlock
	"""
	all_to_send = [var for var in block.inputs if var.peer == dst_block.id]
	all_to_recv = [src for var in dst_block.outputs for src in var if src.peer == block.id]
	for src, dst in zip(all_to_send, all_to_recv):
		grads = src.get(src.to_send, mb_id)
		dst.set(dst.to_process, mb_id, _fake_p2p(grads))
