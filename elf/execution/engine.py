"""
Execution manager
"""

import time
import contextlib
from collections import deque, OrderedDict

import torch
import torch.distributed as dist


from ..scheduling import OperationType, OpOptions
from ..utils import Timer
from .offload import OffloadToCPU, PinnedHostTensorPool

import logging
import os

logger = logging.getLogger("engine")

precise_timings = os.environ.get("ELF_TIMINGS", False)
precise_memory = os.environ.get("ELF_MEMORY", False)


def _fake_p2p(data):
	"""
	Simulates P2P communication by creating a fake work.

	:param data: Input data to be processed
	:type data: Iterator[Tuple[str, Tensor]]
	:return: Tuple with fake communication buffers
	:rtype: Tuple[Tensor]
	"""
	return [None, data.detach()]


def _time_start():
	"""
	Returns a timing event or timestamp to mark the start of an operation.

	:return: A CUDA event if CUDA is available, otherwise a timestamp
	:rtype: torch.cuda.Event or float
	"""
	if torch.cuda.is_available():
		event = torch.cuda.Event(enable_timing=True)
		event.record()
		return event
	else:
		return time.time()


def _time_end(start):
	"""
	Returns a timing event or timestamp to mark the end of an operation.

	:param start: Start time or event
	:type start: torch.cuda.Event or float
	:return: Elapsed time in milliseconds
	:rtype: float
	"""
	if torch.cuda.is_available():
		end = torch.cuda.Event(enable_timing=True)
		end.record()
		end.synchronize()
		return start.elapsed_time(end) / 1000
	else:
		return time.time() - start


def preallocate_pool(pool: PinnedHostTensorPool):
	"""
	Parse the preallocate pool environment variable, and fills the pool accordingly.
	"""
	val = os.environ.get("ELF_PREALLOCATE_POOL")
	if val is None:
		return

	dtypes = {
		"f16": torch.float16,
		"f32": torch.float32,
		"f64": torch.float64,
		"bf16": torch.bfloat16,
		"i8": torch.int8,
		"i16": torch.int16,
		"i32": torch.int32,
	}

	parts = val.split(",")
	for part in parts:
		dtype, size = part.split(":")
		pool.reserve(int(size) * 1024**3, dtype=dtypes[dtype])


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
		if precise_timings and self.rank == 0:
			logger.info("Using precise timings")

		self.id_to_block = {b.id: b for b in self.blocks}
		self.offload_stream = torch.cuda.Stream()

		self.pool = PinnedHostTensorPool()  # global memory pool for this process
		preallocate_pool(self.pool)

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
		- Stats about the execution, as a dict containing:

			- total: total time taken for the execution
			- idle: total time not used for computation for this process
			- start_idle: time between the start of execution and the first computation
			- end_idle: time between the last computation and the end of execution
			- bubble: idle time between first and last computation

		- Detailed stats, as a dict containing:
			- all_events: time taken for each operation
			- memories: total gpu memory allocated after each operation

		.. warning::
			If the environment variable ``ELF_TIMINGS`` is not set, the timings will be wrong.

		.. warning::
			If the environment variable ``ELF_MEMORY`` is not set, the peak memory stats will be wrong (normal allocated memory will still be right).

		:rtype: List[Tensor], List[Tensor], Dict[float], Dict[Dict[Operation, float]]
		"""

		split_batches = [tensor.split(mb_sizes, dim=0) for tensor in batch]
		microbatches = iter(zip(*split_batches))
		if target is not None:
			microtargets = target.split(mb_sizes, dim=0)

		offloaders = {
			id_: [OffloadToCPU(pool=self.pool) for _ in range(len(mb_sizes))] for id_ in self.id_to_block
		}
		# print(f"Rank {self.rank} - Pool size: {self.pool.size()}, offloaders: {offloaders}")

		result = []
		losses = []

		grad_fns = deque()

		if precise_timings:
			dist.barrier()
			torch.cuda.synchronize()
			start = _time_start()

		pipe_start = time.time()
		warmup_time = None
		memories = OrderedDict()
		peak_memories = OrderedDict()

		if precise_memory:
			torch.cuda.reset_peak_memory_stats()

		for op in schedule:
			block = self.id_to_block.get(op.block_id)
			if block is None:
				continue  # not my job

			logger.debug(f"Rank {self.rank} - Executing {op}")

			if profile:
				torch.cuda.nvtx.range_push(f"{block}:{op}")

			if precise_memory:
				torch.cuda.reset_peak_memory_stats()

			if warmup_time is None and op.op != OperationType.RECV_FORWARD:
				# Warmup time is the time spent waiting for the first forward
				# The first operation after that is the end of warmup
				warmup_time = 0
				if precise_timings:
					torch.cuda.synchronize()
					warmup_time = _time_end(start)

			match op.op:
				case OperationType.FORWARD:
					# Use offloader if activation offloading is enabled, otherwise use a dummy context
					if op.options.get(OpOptions.ACTIVATION_OFFLOAD):
						offloader = offloaders[block.id][op.mb_id]
					else:
						offloader = contextlib.nullcontext()

					with offloader:
						y = block.forward(op.mb_id, **op.options)

					# If the block as multiple outputs, this flattens them
					# TODO: correctly handle that in multiple result lists
					if y is not None:
						for output in y:
							if isinstance(output, torch.Tensor):
								result.append(output.detach())
							else:
								logger.warning("Non-tensor output")
								result.append(output)

				case OperationType.BACKWARD_INPUTS:
					block.backward_inputs(op.mb_id, **op.options)

					if op.options.get(OpOptions.ACTIVATION_OFFLOAD):
						offloaders[block.id][op.mb_id].release()

				case OperationType.BACKWARD_PARAMS:
					block.backward_params(op.mb_id, **op.options)

				case OperationType.SEND_FORWARD:
					if op.options.get("dst") in self.id_to_block:
						# The destination block is on the same device ; we bypass p2p comms
						dst_block = self.id_to_block.get(op.options.get("dst"))
						_transfer_forward(block, dst_block, op.mb_id)

					block.send_forward(op.mb_id, **op.options)

				case OperationType.SEND_BACKWARD:
					if op.options.get("dst") in self.id_to_block:
						# The destination block is on the same device ; we bypass p2p comms
						dst_block = self.id_to_block.get(op.options.get("dst"))
						_transfer_backward(block, dst_block, op.mb_id)

					block.send_backward(op.mb_id, **op.options)

				case OperationType.RECV_FORWARD:
					if block.is_first:
						microbatch = next(microbatches)
						for mb, var in zip(microbatch, block.input_variables):
							var.set(var.to_process, op.mb_id, _fake_p2p(mb))

					block.recv_forward(op.mb_id, mb_sizes[op.mb_id], **op.options)

				case OperationType.RECV_BACKWARD:
					block.recv_backward(op.mb_id, mb_sizes[op.mb_id], **op.options)

				case OperationType.LOSS_FORWARD:
					if block.is_last:
						assert op.mb_id < len(result), (
							f"Loss forward for mb {op.mb_id} but only {len(result)} results computed"
						)
						loss, grad_fn = compute_loss(block, result[op.mb_id], microtargets[op.mb_id], loss_fn)
						losses.append(loss)
						grad_fns.append(grad_fn)
						logger.debug(f"Rank {self.rank} - Finished forward of {block}")
					else:
						logger.warning(f"Tried to compute loss on a non-last block {block}")
						continue

				case OperationType.LOSS_BACKWARD:
					if block.is_last:
						assert op.mb_id < len(losses), (
							f"Loss backward for mb {op.mb_id} but only {len(losses)} losses computed"
						)
						grad_fn = grad_fns.popleft()
						with Timer(name=f"backward({block.id}:{op.mb_id})") as timer:
							grads = grad_fn()
						block.compute_time.append(timer)

						# Don't start offloading until we finished computing!
						self.offload_stream.wait_stream(torch.cuda.current_stream())
						with torch.cuda.stream(self.offload_stream):
							with torch.no_grad():
								result[op.mb_id] = result[op.mb_id].to("cpu", non_blocking=True)

						for grad, var in zip(grads, block.output_variables):
							for dst in var:  # should be only one destination
								dst.set(dst.to_process, op.mb_id, _fake_p2p(grad))
					else:
						logger.warning(f"Tried to compute loss backward on a non-last block {block}")
						continue

				case OperationType.RECOMPUTE_FORWARD:
					block.recompute_forward(op.mb_id, **op.options)

				case OperationType.RECOMPUTE_BACKWARD_INPUTS:
					block.recompute_backward_inputs(op.mb_id, **op.options)

				case OperationType.PREFETCH_ACTIVATIONS:
					offloaders[block.id][op.mb_id].prefetch()

				case OperationType.ALL_REDUCE_PARAM_GRADS:
					block.scale_grads(sum(mb_sizes))  # we also average out the gradients here
					block.all_reduce_param_grads(**op.options)

				case _:
					raise Exception(f"Unknown operation : {op}")

			if precise_memory:
				torch.cuda.synchronize()

			memories[str(op)] = torch.cuda.memory_allocated()
			peak_memories[str(op)] = torch.cuda.max_memory_allocated()

			if profile:
				torch.cuda.nvtx.range_pop()

		logger.debug(f"Rank {self.rank} - Finished execution")

		if precise_timings:
			torch.cuda.synchronize()

		cooldown_start = time.time()

		if precise_timings:
			# dist.barrier()
			torch.cuda.synchronize()

		pipe_end = time.time()

		compute_time = 0
		all_events = {}
		if precise_timings:
			for block in self.blocks:
				compute_time += sum([f.time() for f in block.compute_time])
				all_events.update({timer.name: timer.time() for timer in block.compute_time})

		stats = {
			"total": pipe_end - pipe_start,
			"idle": pipe_end - pipe_start - compute_time,
			"start_idle": warmup_time,
			"end_idle": pipe_end - cooldown_start,
		}
		stats["bubble"] = stats["idle"] - stats["start_idle"] - stats["end_idle"]

		detailed_stats = {"all_events": all_events, "memories": memories, "peak_memory": peak_memories}

		for block in self.blocks:
			block.compute_time.clear()

		torch.cuda.current_stream().wait_stream(self.offload_stream)
		for offloaders_list in offloaders.values():
			for offloader in offloaders_list:
				offloader.release()

		return result, losses, stats, detailed_stats


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

	with Timer(name="loss_forward") as timer:
		try:
			loss = loss_fn(output, target, reduction="sum")
		except TypeError:
			loss = loss_fn(output, target)

	block.compute_time.append(timer)
	loss = (
		loss / (target.numel() // target.size(0))
	)  # see documentation for torch losses: loss is reduced by default, but we will divide by batch size at the end (we only have micro-batch size here)

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
	:rtype: None
	"""
	# We match by order and not by name because the names can be different with pre-partitioned models
	all_to_send = [dst for var in block.output_variables for dst in var if dst.peer == dst_block.id]
	all_to_recv = [var for var in dst_block.input_variables if var.peer == block.id]
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
	:rtype: None
	"""
	all_to_send = [var for var in block.input_variables if var.peer == dst_block.id]
	all_to_recv = [src for var in dst_block.output_variables for src in var if src.peer == block.id]
	for src, dst in zip(all_to_send, all_to_recv):
		grads = src.get(src.to_send, mb_id)
		dst.set(dst.to_process, mb_id, _fake_p2p(grads))
