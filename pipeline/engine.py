"""
Execution manager
"""

import torch
import torch.distributed as dist

from collections import deque
import time

from .schedule import OperationType
from .utils import Timer, op_to_str, NameMapping

import logging

logger = logging.getLogger("engine")


def _fake_p2p(data):
	"""
	Simulates P2P communication by creating a fake work.

	:param data: Input data to be processed
	:type data: Iterator[Tuple[str, Tensor]]
	:return: Dictionary with fake communication buffers
	:rtype: dict
	"""
	return {key: [None, value.detach()] for key, value in data}


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
		self.comms = []

	def _run_comms(self):
		"""
		Run all currently batched communications for this device
		Internal function, this should not be used by the user
		"""
		if len(self.comms) == 0:
			return 0

		stream = torch.cuda.Stream()
		with torch.cuda.stream(stream):
			works = dist.batch_isend_irecv(self.comms)
			logger.debug(
				f"Rank {self.rank} - Running batched communications {[op_to_str(c) for c in self.comms]}"
			)
			for w in works:
				w.wait()

		self.comms.clear()
		stream.synchronize()

	def train_step(self, batch, target, loss_fn, schedule, mb_sizes, profile=False):
		"""
		Executes a schedule on a batch of data

		:param batch: input data, only used on the first block of the pipeline
		:type batch: Tensor
		:param target: groundtruth, only used on the last block of the pipeline
		:type target: Tensor
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

		result = []
		losses = []
		current_target = (0, 0)  # micro batch id, position in target tensor

		grad_fns = deque()
		for b in self.blocks:
			b.last_bwd_computed = 0

		dist.barrier()  # useful for timing, but it probably slows down the execution a bit
		pipe_start = time.time()
		warmup_start = None

		for op in schedule:
			block = self.id_to_block.get(op.block_id)
			if block is None:
				continue  # not my job

			logger.debug(f"Computing {op} on block {block} with options {op.options}")

			if profile:
				torch.cuda.nvtx.range_push(f"{block}:{op}")
			if warmup_start is None and op.op != OperationType.RECV_FORWARD:
				# Warmup time is the time spent waiting for the first forward
				# The first operation after that is the end of warmup
				torch.cuda.synchronize()
				warmup_start = time.time()

			if not op.options.get("batched_comm", False):
				self._run_comms()

			match op.op:
				case OperationType.FORWARD:
					y = block.forward(**op.options)
					if y is not None:
						result.append(*y.values())

				case OperationType.BACKWARD:
					block.backward(**op.options)
					block.last_bwd_computed = op.mb_id

				case OperationType.SEND_FORWARD:
					if op.options.get("dst", block.next) == block.rank:
						# The next block is on the same device ; we bypass p2p comms
						next_block = self.id_to_block.get(op.block_id + 1)
						acts = block.act_to_send.popleft()
						acts = {block.name_mapping_out.to_input(k): v for k, v in acts.items()}
						next_block.inputs_to_forward.append(_fake_p2p(acts.items()))

					if comm := block.send_forward(**op.options):
						self.comms.extend(comm)

				case OperationType.SEND_BACKWARD:
					if op.options.get("src", block.previous) == block.rank:
						# The previous block is on the same device ; we bypass p2p comms
						next_block = self.id_to_block.get(op.block_id - 1)
						grads = block.grads_to_send.popleft()
						grads = {block.name_mapping_in.to_output(k): v for k, v in grads.items()}
						next_block.grads_to_backward.append(_fake_p2p(grads.items()))

					if comm := block.send_backward(**op.options):
						self.comms.extend(comm)

				case OperationType.RECV_FORWARD:
					if block.previous is None:
						microbatch = next(microbatches)
						inputs = zip(block.inputs, microbatch)
						block.inputs_to_forward.append(_fake_p2p(inputs))

					if not block.metadata and op.options.get("src", block.previous) == block.rank:
						prev_block = self.id_to_block.get(op.block_id - 1)
						mapping = NameMapping(block.inputs, prev_block.outputs)
						block.name_mapping_in = mapping
						prev_block.name_mapping_out = mapping
						logger.debug(f"Synced metadata of {block} and {prev_block} : {mapping}")

					if comm := block.recv_forward(mb_sizes[op.mb_id], **op.options):
						self.comms.extend(comm)

				case OperationType.RECV_BACKWARD:
					if comm := block.recv_backward(mb_sizes[op.mb_id], **op.options):
						self.comms.extend(comm)

				case OperationType.LOSS_FORWARD:
					if block.next is None:
						i, start = current_target
						end = start + mb_sizes[i]
						loss, grad_fn = compute_loss(block, result[op.mb_id], target[start:end], loss_fn)
						losses.append(loss)
						grad_fns.append(grad_fn)
						logger.debug(f"{block} - Computed loss = {loss.item()}")
						current_target = (i + 1, end)
					else:
						logger.warning(f"Tried to compute loss on a non-last block {block}")
						continue

				case OperationType.LOSS_BACKWARD:
					if block.next is None:
						assert op.mb_id < len(
							losses
						), f"Loss backward for mb {op.mb_id} but only {len(losses)} losses computed"
						grad_fn = grad_fns.popleft()
						with Timer() as timer:
							grads = grad_fn()
						block.compute_time += timer.time()
						block.grads_to_backward.append(_fake_p2p(grads.items()))
					else:
						logger.warning(f"Tried to compute loss backward on a non-last block {block}")
						continue

				case OperationType.ALL_REDUCE_PARAM_GRADS:
					assert block.last_bwd_computed == len(mb_sizes) - 1, (
						"Tried to run all reduce of parameters gradients before the backward for the last microbatch was computed. Last computed backward: "
						+ str(block.last_bwd_computed)
					)
					block.scale_grads(sum(mb_sizes)) # we also average the gradients out here
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
			compute_time += block.compute_time
			block.compute_time = 0

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
	:rtype: Tensor, Callable[[], Dict[str, Tensor]]
	"""
	output = output.detach().requires_grad_()
	if len(block.outputs) != 1:
		raise RuntimeError("Multiple outputs not supported yet for loss computation")

	with Timer() as timer:
		try:
			loss = loss_fn(output, target, reduction="sum")
		except TypeError:
			loss = loss_fn(output, target)

	block.compute_time += timer.time()

	def grad_fn():
		loss.backward()
		return { block.outputs[0]: output.grad.data }

	return loss, grad_fn
