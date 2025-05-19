import copy
import os
import sys

sys.path.append(".")

import torch
import torch.distributed as dist

from models.simple import FullTransformer
from elf.zb_utils import LayerDW
from elf import Pipeline, get_sources_targets_sequential


def bench(model, parts, scheduler, placement, gradient_accumulation=1):
	local_rank = int(os.getenv("LOCAL_RANK"))
	world_size = dist.get_world_size()
	rank = dist.get_rank()
	n_iterations = 20

	microbatch_size = 2 // gradient_accumulation
	n_micro_batches = world_size * 2 * gradient_accumulation
	batch_size = microbatch_size * n_micro_batches

	loss_fn = model.loss_fn

	# Create pipeline
	sources, dsts = get_sources_targets_sequential(placement)
	pipe = Pipeline(
		parts,
		None,
		placement=placement,
		partitioner=False,
		schedule=scheduler,
		sources=sources,
		targets=dsts,
	)

	# test_correctness(model, pipe)

	inputs = model.get_sample(batch_size)
	targets = model.get_target(batch_size)

	# Warmup iterations
	for _ in range(3):
		_ = pipe(inputs.clone(), targets.clone(), loss_fn, split_size=microbatch_size)

	del inputs, targets

	torch.cuda.synchronize()
	torch.cuda.empty_cache()
	dist.barrier()

	# Time n iterations
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	torch.cuda.reset_peak_memory_stats()

	start.record()
	for i in range(n_iterations):
		model.zero_grad()
		inputs = model.get_sample(batch_size)
		targets = model.get_target(batch_size)
		_ = pipe(inputs, targets, loss_fn, split_size=microbatch_size)

	dist.barrier()
	torch.cuda.synchronize()
	end.record()
	peak_mem = torch.cuda.max_memory_allocated() / (2**30)

	iter_time = start.elapsed_time(end) / (n_iterations * 1000)

	all_peak_mems = (
		[torch.tensor(0.0, device=local_rank) for _ in range(world_size)] if rank == 0 else None
	)
	dist.gather(torch.tensor(peak_mem, device=local_rank), all_peak_mems, dst=0)

	pipe.clear()
	del inputs, targets
	return iter_time, all_peak_mems


def get_handcrafted_imbalanced_partition(model, rank, placement, factors):
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

	return [parts[i] for i, p in enumerate(placement) if p == rank]


def find(sched, optype, block_id, mb_id):
	for i, op in enumerate(sched):
		if op.mb_id == mb_id and op.op == optype and op.block_id == block_id:
			return i
	return None


def test_correctness(model, pipe):
	rank = dist.get_rank()
	world_size = dist.get_world_size()

	# Sync parameters on CPU
	for param in model.parameters():
		tensor = param.data.cuda()
		dist.broadcast(tensor, src=0)
		param.data = tensor.cpu()

	ref_model = copy.deepcopy(model)

	# Move back all necessary params to GPU
	for block in pipe.blocks:
		block.model.cuda()

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
	ref_model.eval()

	# Execute using pipeline
	pipe.zero_grad()
	y, loss = pipe(inputs, targets, loss_fn, split_size=1)

	all_grads = {}
	for name, param in pipe.blocks[0].model.named_parameters():
		if param.grad is not None:
			all_grads[name] = param.grad.cpu()

	last = world_size - 1
	if rank == last:
		ref_model.cpu()  # doesnt fit in gpu memory
		ref_model.zero_grad()
		y_ref = ref_model(inputs)
		loss_ref = loss_fn(y_ref, targets)

		y = y.cpu()
		loss = loss.cpu()

		assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), (
			f"y mismatch: {torch.norm(y - y_ref)} ({100 * torch.norm(y - y_ref) / torch.norm(y_ref)}%)"
		)

		assert torch.allclose(loss, loss_ref, atol=atol, rtol=rtol), (
			f"loss mismatch: {torch.norm(loss - loss_ref)} ({100 * torch.norm(loss - loss_ref) / torch.norm(loss_ref)}%)"
		)

		loss_ref.backward()
		for module in ref_model.modules():
			if isinstance(module, LayerDW):
				module.move_last_computed("input", 0)
				module.move_last_computed("grad_output", 0)
				module.backward(0)

		offset = 0

		def rename_param(name):
			splitted = name.split(".")
			splitted[0] = "block_" + str(offset + int(splitted[0]))
			return ".".join(splitted)

		all_grads_ref = {
			name: param.grad for name, param in ref_model.named_parameters() if param.grad is not None
		}

		for peer in range(0, world_size - 1):
			grads = [{}]
			dist.recv_object_list(grads, peer)
			for name, grad in grads[0].items():
				name = rename_param(name)
				assert torch.allclose(grad.cpu(), all_grads_ref[name], atol=atol, rtol=rtol), (
					f"grad mismatch ({name}): {torch.norm(grad.cpu() - all_grads_ref[name])} ({100 * torch.norm(grad.cpu() - all_grads_ref[name]) / torch.norm(all_grads_ref[name])}%)"
				)
				del all_grads_ref[name]
			offset += max(int(name.split(".")[0]) for name in grads[0].keys()) + 1
			del grads

		for name, grad in all_grads.items():
			name = rename_param(name)
			assert torch.allclose(grad, all_grads_ref[name], atol=atol, rtol=rtol), (
				f"grad mismatch ({name}): {torch.norm(grad - all_grads_ref[name])} ({100 * torch.norm(grad - all_grads_ref[name]) / torch.norm(all_grads_ref[name])}%)"
			)
	else:
		dist.send_object_list([all_grads], last)

	pipe.blocks[0].model.cuda()  # move it back to gpu as it was before
	model.train()
