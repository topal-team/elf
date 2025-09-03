import os
import time
import copy

from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist

from elf import Pipeline, get_sources_targets_sequential
from elf.zb_utils import LayerDW
from elf.registry import SCHEDULERS, resolve
from elf.scheduling.scheduling import Operation, OpOptions, OperationType, compute_types
from models.simple import Attention, FullTransformer, TransformerBlock


def init_dist(backend="nccl"):
	"""
	Initialize the distributed environment using the given backend.
	Works with torchrun, mpirun, or srun.

	Returns:
		local_rank: int
		rank: int
		world_size: int
	"""

	# Either started with torchrun, mpirun, or srun
	local_rank = int(
		os.getenv("LOCAL_RANK", os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", os.getenv("SLURM_LOCALID")))
	)
	torch.cuda.set_device(local_rank)

	dist.init_process_group(backend=backend, device_id=torch.device(f"cuda:{local_rank}"))

	return local_rank, dist.get_rank(), dist.get_world_size()


def bench(model, parts, scheduler, placement, dtype=torch.float32):
	local_rank = int(os.getenv("LOCAL_RANK"))
	world_size = dist.get_world_size()
	rank = dist.get_rank()
	n_iterations = 30

	microbatch_size = 1
	n_micro_batches = world_size * 2
	batch_size = microbatch_size * n_micro_batches

	loss_fn = model.loss_fn

	# Create pipeline
	sources, dsts = get_sources_targets_sequential(placement)
	pipe = Pipeline(
		parts,
		None,
		placement=placement,
		partitioner=False,
		scheduler=scheduler,
		sources=sources,
		targets=dsts,
	)

	# test_correctness(model, pipe)

	# Warmup iterations
	for _ in range(3):
		_ = pipe(
			model.get_sample(batch_size, dtype, device="cuda"),
			model.get_target(batch_size, dtype, device="cuda"),
			loss_fn,
			split_size=microbatch_size,
		)

	torch.cuda.synchronize()
	dist.barrier()
	torch.cuda.reset_peak_memory_stats()

	start_time = time.time()

	# Time n iterations
	for i in range(n_iterations):
		pipe.zero_grad()
		_ = pipe(
			model.get_sample(batch_size, dtype, device="cuda"),
			model.get_target(batch_size, dtype, device="cuda"),
			loss_fn,
			split_size=microbatch_size,
		)

	torch.cuda.synchronize()
	dist.barrier()
	peak_mem = torch.cuda.max_memory_allocated() / (2**30)

	iter_time = (time.time() - start_time) / n_iterations

	all_peak_mems = (
		[torch.tensor(0.0, device=local_rank) for _ in range(world_size)] if rank == 0 else None
	)
	dist.gather(torch.tensor(peak_mem, device=local_rank), all_peak_mems, dst=0)

	pipe.clear()
	return iter_time, all_peak_mems


def meta_to_device(model, device="cuda"):
	model.to_empty(device=device)
	for param in model.parameters():
		if hasattr(param, "reset_parameters"):
			param.reset_parameters()

	return model


def balanced_partition(n: int, placement: List[int]) -> List[int]:
	"""Computes the number of blocks per GPU to balance the load."""
	if n < len(placement):
		print(f"n = {n} is less than the number of GPUs = {len(placement)}")

	blocks_per_gpu = n // len(placement)
	remainder = n % len(placement)
	return [blocks_per_gpu + (1 if i < remainder else 0) for i in range(len(placement))]


def get_handcrafted_imbalanced_partition(model, rank, placement, factors):
	"""Get blocks associated to the current rank, on GPU, so that the load follows `factors`."""
	num_blocks = len(model.blocks)
	num_ranks = len(placement)
	parts = [None] * num_ranks
	assert int(sum(factors)) == int(num_blocks), (
		f"Sum of factors ({sum(factors)}) does not equal number of blocks ({num_blocks})"
	)

	start_idx = 0
	for i in range(num_ranks):
		end_idx = start_idx + factors[i]

		if isinstance(model, FullTransformer) and i == 0:
			parts[i] = torch.nn.Sequential(model.embed, *model.blocks[start_idx:end_idx])
		else:
			parts[i] = torch.nn.Sequential(*model.blocks[start_idx:end_idx])

		if isinstance(model, FullTransformer) and i == num_ranks - 1:
			parts[i].append(model.head)  # doesn't work for multi waves

		start_idx = end_idx

	parts = [parts[i] for i, p in enumerate(placement) if p == rank]
	parts = [meta_to_device(p, "cuda") for p in parts]
	return parts


def get_handcrafted_partition(model, rank, placement):
	"""Get blocks associated to the current rank, on GPU, so that the load is balanced."""
	factors = balanced_partition(len(model.blocks), placement)
	return get_handcrafted_imbalanced_partition(model, rank, placement, factors)


def find(sched, optype, block_id, mb_id):
	for i, op in enumerate(sched):
		if op.mb_id == mb_id and op.op == optype and op.block_id == block_id:
			return i
	return None


def test_correctness(model, pipe):
	rank = dist.get_rank()
	world_size = dist.get_world_size()

	atol = 1e-3
	rtol = 1e-3
	inputs = model.get_sample(world_size * 2).cuda()
	dist.broadcast(inputs, src=0)
	targets = model.get_target(world_size * 2).cuda()
	dist.broadcast(targets, src=0)
	inputs = inputs.cpu()
	targets = targets.cpu()

	loss_fn = model.loss_fn
	model.eval()  # disable dropout

	# Execute using pipeline
	pipe.zero_grad()
	y, loss = pipe(inputs, targets, loss_fn, split_size=1)

	all_grads = {}
	for block in pipe.blocks:
		for name, param in block.model.named_parameters():
			if param.grad is not None:
				all_grads[name] = param.grad.cpu()

	last = world_size - 1
	all_params = pipe.gather_parameters(dst=last)
	if rank == last:
		ref_model = copy.deepcopy(model)
		meta_to_device(ref_model, "cpu")
		ref_model.load_state_dict(all_params)
		ref_model.eval()
		ref_model.zero_grad()
		y_ref = ref_model(inputs)
		loss_ref = loss_fn(y_ref, targets)

		y = y.cpu()
		loss = loss.cpu()

		assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
			f"y mismatch: {torch.norm(y - y_ref)} ({100 * torch.norm(y - y_ref) / torch.norm(y_ref)}%) ({torch.norm(y)} vs {torch.norm(y_ref)})"
		)

		assert torch.allclose(loss, loss_ref, atol=atol, rtol=rtol), (
			f"loss mismatch: {torch.norm(loss - loss_ref)} ({100 * torch.norm(loss - loss_ref) / torch.norm(loss_ref)}%) ({torch.norm(loss)} vs {torch.norm(loss_ref)})"
		)

		loss_ref.backward()
		for module in ref_model.modules():
			if isinstance(module, LayerDW):
				module.move_last_computed("input", 0)
				module.move_last_computed("grad_output", 0)
				module.backward(0)

		all_grads_ref = {
			name: param.grad for name, param in ref_model.named_parameters() if param.grad is not None
		}

		for peer in range(0, world_size - 1):
			grads = [{}]
			dist.recv_object_list(grads, peer)
			for name, grad in grads[0].items():
				assert torch.allclose(grad.cpu(), all_grads_ref[name], atol=atol, rtol=rtol), (
					f"grad mismatch ({name}): {torch.norm(grad.cpu() - all_grads_ref[name])} ({100 * torch.norm(grad.cpu() - all_grads_ref[name]) / torch.norm(all_grads_ref[name])}%)"
				)
				del all_grads_ref[name]
			del grads

		for name, grad in all_grads.items():
			assert torch.allclose(grad, all_grads_ref[name], atol=atol, rtol=rtol), (
				f"grad mismatch ({name}): {torch.norm(grad - all_grads_ref[name])} ({100 * torch.norm(grad - all_grads_ref[name]) / torch.norm(all_grads_ref[name])}%)"
			)
	else:
		dist.send_object_list([all_grads], last)

	pipe.blocks[0].model.cuda()  # move it back to gpu as it was before
	model.train()


def get_checkpointed_scheduler(scheduler, type):
	"""
	Get a checkpointed scheduler.

	Args:
		scheduler: The scheduler to checkpoint.
		type: The type of checkpointing to use.
			- "full": Checkpoint all operations.
			- "simple": Checkpoint only operations that are simple modules (nn.GELU, nn.LayerNorm, nn.Dropout).

	Returns:
		A checkpointed scheduler.
	"""
	scheduler = resolve(scheduler, SCHEDULERS)
	if type == "full":

		def checkpoint(name, module):
			return isinstance(module, TransformerBlock)
	elif type == "simple":

		def checkpoint(name, module):
			return isinstance(module, (nn.GELU, nn.LayerNorm, nn.Dropout))

	elif type == "selective":

		def checkpoint(name, module):
			return isinstance(module, Attention)

	else:
		raise ValueError(f"Invalid checkpointing type: {type}")

	def checkpointed_scheduler(*args, **kwargs):
		schedule = scheduler(*args, **kwargs)
		for op in schedule:
			if op.op == OperationType.FORWARD:
				op.options[OpOptions.REMAT_STRATEGY] = checkpoint

		return schedule

	return checkpointed_scheduler


def get_offloaded_scheduler(scheduler, ratio, prefetching_time=1):
	"""
	Get an offloaded scheduler.

	Args:
		scheduler: The scheduler to offload.
		ratio: The ratio of activations to offload.
	"""
	scheduler = resolve(scheduler, SCHEDULERS)

	def offloaded_scheduler(*args, **kwargs):
		schedule = scheduler(*args, **kwargs)

		placement = args[0]
		n_devices = max(placement) + 1

		offloaded_mbs = [[] for _ in range(n_devices)]
		for op in schedule:
			if op.op == OperationType.FORWARD and op.mb_id % ratio == 0:
				op.options[OpOptions.ACTIVATION_OFFLOAD] = True
				offloaded_mbs[op.rank].append((op.mb_id, op.block_id))

		# Start prefetching during a previous computation ("ratio" computations before the backward)
		for rank in range(n_devices):
			backward_op = None
			for mb, block_id in offloaded_mbs[rank]:
				backward_inputs_idx = None
				for i, op in enumerate(schedule):
					if op.block_id == block_id and op.mb_id == mb and op.op == OperationType.BACKWARD_INPUTS:
						backward_inputs_idx = i
						backward_op = op
						backward_op.options[OpOptions.ACTIVATION_OFFLOAD] = True
						break

				# Count backward from BACKWARD_INPUTS to find kth computation
				computation_count = 0
				for i in range(backward_inputs_idx - 1, -1, -1):
					op = schedule[i]
					if op.rank == rank and op.op in compute_types and op.op != OperationType.BACKWARD_PARAMS: # ZB not supported yet
						computation_count += 1
						if op.op == OperationType.FORWARD and op.mb_id == mb and op.block_id == block_id:
							# This is the offloaded forward ; there is no time to prefetch
							# We just disable offloading for this forward
							op.options[OpOptions.ACTIVATION_OFFLOAD] = False
							backward_op.options[OpOptions.ACTIVATION_OFFLOAD] = False
							break

						if computation_count == prefetching_time:
							# If last op is the offloaded forward, skip prefetching
							for j in range(i - 1, -1, -1):
								if schedule[j].rank == rank and schedule[j].op in compute_types:
									if schedule[j].op == OperationType.FORWARD and schedule[j].mb_id == mb and schedule[j].block_id == block_id:
										schedule[j].options[OpOptions.ACTIVATION_OFFLOAD] = False
										backward_op.options[OpOptions.ACTIVATION_OFFLOAD] = True
									break

							# Otherwise, add prefetch operation before this computation
							prefetch_op = Operation(block_id, mb, OperationType.PREFETCH_ACTIVATIONS, rank)
							schedule.insert(i, prefetch_op)
							break

		return schedule

	return offloaded_scheduler
