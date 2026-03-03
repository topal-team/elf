"""ILP-based rematerialization optimization for pipeline schedules.

Profiles pipeline stages, solves an integer linear program to decide which
operations to recompute, and wraps the scheduler to apply those decisions.

Usage from Pipeline:
	Set ``memory_budget`` (MB) in PipelineConfig.  The Pipeline will
	automatically profile stages at partition time and solve the ILP lazily
	on the first training step (once the number of micro-batches is known).

For more details, see: Adrien Aguila--Multner, Olivier Beaumont, Lionel Eyraud-Dubois, Julia Gusak. Optimized Forward-Backward Rematerialization for Memory-Efficient Pipeline Parallel Training. 2025. https://inria.hal.science/hal-05151601
"""

import time
import logging
import numpy as np

from dataclasses import dataclass

import torch
import torch.distributed as dist

from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.checkpoint import checkpoint

from elf.utils import TensorMetadata, Timer
from elf.zb_utils import LayerDW
from elf.partitioners.utils import (
	Signature,
	get_sources_targets_sequential,
	signatures_from_sources_targets,
)
from elf.scheduling.scheduling import OpOptions, OperationType, Operation

logger = logging.getLogger("elf.ilp")


# ── Profiling ────────────────────────────────────────────────────────────────


def _bparams(stage: torch.nn.Module):
	for module in stage.modules():
		if isinstance(module, LayerDW):
			module.move_last_computed("input", 0)
			module.move_last_computed("grad_output", 0)
			module.backward(0)


def _fake_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
	return sum(o.sum() for o in outputs if isinstance(o, torch.Tensor))


def _measure_full_pass(
	stage: torch.nn.Module, sample: TensorMetadata, n_iter: int = 3
) -> tuple[dict, dict, dict, float, float, torch.Tensor]:
	"""Measure T, M, Mpeak for F/B/W, input/output sizes, and warmup output.

	The warmup output (detached, on CPU) is returned so that it can be used
	as the input sample for the next pipeline stage, avoiding an extra forward
	pass just for shape propagation.
	"""
	outputs = stage(sample)
	outputs = (outputs,) if not isinstance(outputs, tuple) else outputs
	warmup_output = outputs[0].detach().cpu()
	loss = _fake_loss(outputs)
	loss.backward()
	_bparams(stage)
	del outputs, loss
	torch.cuda.synchronize()

	times = {"f": [], "b": [], "w": []}
	mems = {"f": [], "b": [], "w": []}
	peaks = {"f": [], "b": [], "w": []}

	for _ in range(n_iter):
		torch.cuda.reset_peak_memory_stats()
		start_mem = torch.cuda.memory_allocated()
		with Timer() as timer:
			outputs = stage(sample)
			outputs = (outputs,) if not isinstance(outputs, tuple) else outputs
			loss = _fake_loss(outputs)

		minput = (sample.nbytes) / 1024 / 1024
		moutput = (
			sum(output.nbytes for output in outputs if isinstance(output, torch.Tensor)) / 1024 / 1024
		)

		times["f"].append(timer.time() * 1000)  # in ms
		mems["f"].append((torch.cuda.memory_allocated() - start_mem) / 1024 / 1024)
		peaks["f"].append((torch.cuda.max_memory_allocated() - start_mem) / 1024 / 1024)

		before_b = torch.cuda.memory_allocated()
		with Timer() as timer:
			loss.backward()
			del outputs, loss

		times["b"].append(timer.time() * 1000)  # in ms
		mems["b"].append((torch.cuda.memory_allocated() - start_mem) / 1024 / 1024)
		peaks["b"].append((torch.cuda.max_memory_allocated() - before_b) / 1024 / 1024)

		before_w = torch.cuda.memory_allocated()
		with Timer() as timer:
			_bparams(stage)

		times["w"].append(timer.time() * 1000)  # in ms
		mems["w"].append((torch.cuda.memory_allocated() - start_mem) / 1024 / 1024)
		peaks["w"].append((torch.cuda.max_memory_allocated() - before_w) / 1024 / 1024)

	avg_times = {k: np.median(times[k]).item() for k in times}
	avg_mems = {k: np.median(mems[k]).item() for k in mems}
	avg_peaks = {k: np.median(peaks[k]).item() for k in peaks}
	return avg_times, avg_mems, avg_peaks, minput, moutput, warmup_output


def _measure_ckpt_forward(stage: torch.nn.Module, sample: torch.Tensor) -> tuple[float, float]:
	torch.cuda.reset_peak_memory_stats()
	start_mem = torch.cuda.memory_allocated()
	_ = checkpoint(stage, sample, use_reentrant=False)
	peak_ckpt = (torch.cuda.max_memory_allocated() - start_mem) / 1024 / 1024

	mba = 0
	for module in stage.modules():
		if isinstance(module, LayerDW):
			inputs = getattr(module, "last_input", None)
			if inputs is not None:
				mba += inputs.nbytes
			delattr(module, "last_input")

	return peak_ckpt, mba


def _measure_communication(
	signature: Signature, minput: float, moutput: float, placement: list[int], n_iter=100, group=None
) -> float:
	rank = dist.get_rank() if group is None else dist.get_rank(group)
	for source in signature.sources:
		if source is None or placement[source] == rank:
			continue
		for _ in range(n_iter):
			buffer = torch.empty(int(minput) * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
			dist.recv(buffer, placement[source], group=group)

	torch.cuda.synchronize()
	times = []
	for target in signature.targets:
		assert target is None or len(target) == 1, "Only one target is supported with ILP for now"
		target = target[0] if target is not None else None
		if target is None or placement[target] == rank:
			times.append(0.0)
			continue
		for _ in range(n_iter):
			buffer = torch.empty(int(moutput) * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
			with Timer() as timer:
				dist.send(buffer, placement[target], group=group)
			times.append(timer.time() * 1000)  # in ms

	tcomm = np.median(times).item()
	return tcomm


def profile(
	stages: list[torch.nn.Module],
	samples: list[torch.Tensor],
	signatures: list[Signature],
	placement: list[int],
) -> list[dict]:
	"""Measure profiling statistics for each stage (standalone, with communication).

	Statistics format per stage::

		{
			"T": {"f": float, "b": float, "w": float},
			"M": {"f": float, "b": float, "w": float},
			"Mpeak": {"f": float, "b": float, "w": float},
			"Mpeak_ckpt": float,
			"Mparams": float,
			"Mba": float,
			"Mbg": float,
			"Minput": float,
			"Moutput": float,
			"Tcomm": float,
			"forward_remat_options": [...],
			"backward_remat_options": [...],
		}
	"""
	_KEYS = (
		"T",
		"M",
		"Mpeak",
		"Mpeak_ckpt",
		"Mparams",
		"Mba",
		"Mbg",
		"Minput",
		"Moutput",
		"Tcomm",
		"forward_remat_options",
		"backward_remat_options",
	)

	statistics = [{} for _ in stages]

	for stage, stats, sample, signature in zip(stages, statistics, samples, signatures):
		T, M, Mpeak, Minput, Moutput, _ = _measure_full_pass(stage, sample)
		peak_ckpt, mba = _measure_ckpt_forward(stage, sample)
		mbg = M["b"] - mba
		tcomm = _measure_communication(signature, Minput, Moutput, placement)

		stats["T"] = T
		stats["M"] = M
		stats["Mpeak"] = Mpeak
		stats["Mpeak_ckpt"] = peak_ckpt
		stats["Mba"] = mba
		stats["Mbg"] = mbg
		stats["Minput"] = Minput
		stats["Moutput"] = Moutput
		stats["Mparams"] = sum(p.nbytes for p in stage.parameters()) / 1024 / 1024
		stats["Tcomm"] = tcomm
		stats["forward_remat_options"] = []
		stats["backward_remat_options"] = []

	for stats in statistics:
		for key in _KEYS:
			assert key in stats, f"Key {key} not found in stats"

	return statistics


def init_tensor_(tensor: torch.Tensor):
	"""Fill tensor in-place with random data appropriate for its dtype.
	.. warning:: For integer dtype, the tensor is filled with random integers between 0 and 100. If you have an embedding with dim < 100, this will cause an error.
	"""
	match tensor.dtype:
		case torch.float32 | torch.float16 | torch.bfloat16:
			tensor.normal_(mean=0.0, std=1.0)
		case torch.int32 | torch.int64 | torch.int8 | torch.int16:
			tensor.random_(0, 100)
		case torch.bool:
			tensor.random_(0, 2)
		case _:
			raise ValueError(f"Unsupported dtype: {tensor.dtype}")


def profile_stage(
	stage: torch.nn.Module, sample_metadata: TensorMetadata, n_iter: int = 10
) -> tuple[dict, torch.Tensor]:
	"""Profile a single stage for Pipeline integration (no communication).

	:return: ``(stats_dict, next_stage_input)`` where ``next_stage_input``
		is the detached CPU output from the warmup forward, ready to be
		used as input for the next pipeline stage.
	"""
	sample = sample_metadata.get_buffer(1)  # TODO: use the actual mb size instead of 1!
	init_tensor_(sample)
	# Or, we could also profile for 1 and assume everything is linear in batch size ; then we need to multiply everything by mb size before solving
	T, M, Mpeak, Minput, Moutput, next_input = _measure_full_pass(stage, sample, n_iter)
	peak_ckpt, mba = _measure_ckpt_forward(stage, sample)

	stats = {
		"T": T,
		"M": M,
		"Mpeak": Mpeak,
		"Mpeak_ckpt": peak_ckpt,
		"Mparams": sum(p.nbytes for p in stage.parameters()) / 1024 / 1024,
		"Mba": mba,
		"Mbg": M["b"] - mba,
		"Minput": Minput,
		"Moutput": Moutput,
		"Tcomm": 0.0,
		"forward_remat_options": [],
		"backward_remat_options": [],
	}
	return stats, next_input


# ── Distributed profiling ────────────────────────────────────────────────────


def propagate_sample(block_models, placement, sample, pp_group):
	"""Determine per-block input shapes without running real computation.

	Uses ``FakeTensorMode`` to trace shapes through each block.  At rank
	boundaries only a :class:`TensorMetadata` descriptor (16 floats) is
	exchanged via GPU p2p.

	:param block_models: dict mapping block id -> nn.Module (only for blocks on this rank)
	:param placement: stage-to-rank mapping
	:param sample: original model input (on any device)
	:param pp_group: pipeline-parallel process group
	:return: dict mapping local block ids to :class:`TensorMetadata` for each block's input.
	"""
	rank = dist.get_rank()
	n_blocks = len(placement)

	block_in_metas = {}
	out_meta = None
	mb_size = sample.size(0) if isinstance(sample, torch.Tensor) else sample[0].size(0)
	fake_mode = FakeTensorMode(
		allow_non_fake_inputs=True
	)  # necessary unless we also convert all parameters
	converter = fake_mode.fake_tensor_converter
	fake_sample = converter.from_real_tensor(fake_mode, sample)

	for block_id in range(n_blocks):
		owner = placement[block_id]
		prev_owner = placement[block_id - 1] if block_id > 0 else None

		if owner == rank:
			if block_id == 0:
				ref = sample[0] if isinstance(sample, (list, tuple)) else sample
				in_meta = TensorMetadata(ref[0])
			elif prev_owner == rank:
				in_meta = out_meta
			else:
				meta_buf = torch.empty(TensorMetadata.MAX_SIZE, device="cuda")
				dist.recv(meta_buf, src=prev_owner, group=pp_group)
				in_meta = TensorMetadata.from_tensor(meta_buf)

			with fake_mode, torch.no_grad():
				fake_inp = in_meta.get_buffer(mb_size) if block_id != 0 else fake_sample
				block_in_metas[block_id] = in_meta
				fake_out = block_models[block_id](fake_inp)
				if isinstance(fake_out, tuple):
					fake_out = fake_out[0]
				assert fake_out.size(0) == mb_size, (
					f"Fake output has size {fake_out.size(0)} but expected {mb_size}"
				)
				out_meta = TensorMetadata(fake_out[0])

			for module in block_models[block_id].modules():
				if isinstance(module, LayerDW):
					module.clear()

		elif prev_owner == rank:
			dist.send(out_meta.to_tensor(), dst=owner, group=pp_group)

	return block_in_metas


def profile_communication(local_stats, signatures, placement, pp_group):
	"""Measure inter-rank communication time for each local block.

	All ranks call :func:`_measure_communication` for their own blocks
	simultaneously, so the sends and receives across ranks match naturally.

	Mutates *local_stats* in-place (sets ``"Tcomm"`` for each local block).

	:param local_stats: dict mapping local block id -> stats dict
	:param signatures: list of :class:`Signature` for all blocks
	:param placement: stage-to-rank mapping
	:param pp_group: pipeline-parallel process group
	"""
	for block_id, stats in local_stats.items():
		stats["Tcomm"] = _measure_communication(
			signatures[block_id], stats["Minput"], stats["Moutput"], list(placement), group=pp_group
		)


def profile_all_stages(blocks, signatures, placement, sample, pp_group):
	"""Profile pipeline stages in parallel across all PP ranks.

	1. :func:`propagate_sample` determines per-block input shapes.
	2. Each rank profiles its own blocks via :func:`profile_stage`.
	3. :func:`profile_communication` measures inter-rank transfer times.
	4. Stats are gathered across the PP group with ``all_gather_object``.

	:param blocks: list of :class:`~elf.execution.PipelineBlock` on this rank
	:param signatures: list of :class:`Signature` for all blocks
	:param placement: stage-to-rank mapping
	:param sample: original model input
	:param pp_group: pipeline-parallel process group
	:return: ordered list of per-stage statistics dicts (length = n_blocks)
	"""
	rank = dist.get_rank()
	if rank == 0:
		logger.info("Profiling stages for ILP..")

	block_models = {b.id: b.model for b in blocks}
	block_in_metas = propagate_sample(block_models, placement, sample, pp_group)

	local_stats = {}
	for block in sorted(blocks, key=lambda b: b.id):
		stats, _ = profile_stage(block.model, block_in_metas[block.id])
		block.model.zero_grad(set_to_none=True)
		for module in block.model.modules():
			if isinstance(module, LayerDW):
				module.clear()
		torch.cuda.empty_cache()
		local_stats[block.id] = stats

	profile_communication(local_stats, signatures, placement, pp_group)

	all_stats = [None] * dist.get_world_size(pp_group)
	dist.all_gather_object(all_stats, local_stats, group=pp_group)

	statistics = [None] * len(placement)
	for rank_stats in all_stats:
		if rank_stats is not None:
			for block_id, s in rank_stats.items():
				statistics[block_id] = s

	return statistics


# ── ILP data structures ──────────────────────────────────────────────────────


@dataclass
class RematOption:
	name: str
	type: str  # "forward" or "backward"
	mem_freed: float
	time_overhead: float


def _op_name_to_index(op):
	return ["f", "b", "w"].index(op)


def _generate_simplified_schedule(placement, n_micro_batches, scheduler):
	"""Generate per-rank schedule as ``[rank] -> [(op, block_id, mb_id), ...]``.

	*scheduler* is a resolved callable, not a registry key.
	"""
	p = len(set(placement))
	signatures = signatures_from_sources_targets(*get_sources_targets_sequential(placement))
	schedule = scheduler(placement, n_micro_batches, signatures)

	simplified = [[] for _ in range(p)]
	for op in schedule:
		match op.op:
			case OperationType.FORWARD:
				simplified[op.rank].append(("f", op.block_id, op.mb_id))
			case OperationType.BACKWARD_INPUTS:
				simplified[op.rank].append(("b", op.block_id, op.mb_id))
			case OperationType.BACKWARD_PARAMS:
				simplified[op.rank].append(("w", op.block_id, op.mb_id))
	return simplified


def _scale_stats_by_batch_size(statistics, mb_size):
	"""Scale the statistics by the micro-batch size."""
	return [
		{
			"T": {k: v * mb_size for k, v in stats["T"].items()},
			"M": {k: v * mb_size for k, v in stats["M"].items()},
			"Mpeak": {k: v * mb_size for k, v in stats["Mpeak"].items()},
			"Mpeak_ckpt": stats["Mpeak_ckpt"] * mb_size,
			"Mparams": stats["Mparams"],  # this does not depend on the batch size
			"Minput": stats["Minput"] * mb_size,
			"Moutput": stats["Moutput"] * mb_size,
			"Tcomm": stats["Tcomm"] * mb_size,
			"forward_remat_options": stats["forward_remat_options"],
			"backward_remat_options": stats["backward_remat_options"],
		}
		for stats in statistics
	]


def build_ilp_params(statistics, placement, scheduler, n_micro_batches, memory_budget, mb_size):
	"""Convert profiling stats to the params dict expected by the ILP solver."""
	p = len(set(placement))
	nstages = len(placement)
	assert len(statistics) == nstages

	sched = _generate_simplified_schedule(placement, n_micro_batches, scheduler)
	scaled_stats = _scale_stats_by_batch_size(statistics, mb_size)
	assert all((key in scaled for key in stats) for scaled, stats in zip(scaled_stats, statistics)), (
		"Scaled stats do not contain all keys of the original stats"
	)

	stages = []
	for stats in scaled_stats:
		T = stats["T"]
		M = stats["M"]
		Mpeak = dict(stats["Mpeak"])
		Mpeak["f_no_grad"] = stats["Mpeak_ckpt"]
		Minput = stats["Minput"]
		Moutput = stats["Moutput"]

		fwd_remat_options = [
			RematOption(opt["name"], "forward", opt["mem_freed"], opt["overhead"])
			for opt in stats["forward_remat_options"]
		]
		fwd_remat_options.append(RematOption("full_fwd", "forward", M["f"] - Minput - Moutput, T["f"]))

		bwd_remat_options = [
			RematOption(opt["name"], "backward", opt["mem_freed"], opt["overhead"])
			for opt in stats["backward_remat_options"]
		]
		bwd_remat_options.append(
			RematOption("full_bwd", "backward", M["b"] - Moutput - Minput, T["f"] + T["b"])
		)

		stages.append(
			{
				"T": T,
				"M": M,
				"Mpeak": Mpeak,
				"Tcomm": stats["Tcomm"],
				"Mparams": stats["Mparams"],
				"fwd_remat_options": fwd_remat_options,
				"bwd_remat_options": bwd_remat_options,
			}
		)

	return {
		"p": p,
		"m": n_micro_batches,
		"nstages": nstages,
		"MGpu": memory_budget,
		"sched": sched,
		"placement": list(placement),
		"stages": stages,
	}


# ── ILP solver ───────────────────────────────────────────────────────────────


def solve_remat(
	statistics: list[dict],
	placement: list[int],
	scheduler,
	n_micro_batches: int,
	memory_budget: int,
	mb_sizes: list[int],
	time_limit: int = 60,
) -> dict:
	"""Solve the rematerialization ILP.

	Given profiled statistics for each stage, finds the per-(stage, microbatch)
	remat strategy that minimizes iteration time while keeping peak memory
	within the given budget.

	:param statistics: per-stage profiling dicts from :func:`profile` or :func:`profile_stage`
	:param placement: stage-to-rank mapping
	:param scheduler: resolved scheduler callable
	:param n_micro_batches: number of micro-batches
	:param memory_budget: GPU memory budget in MB
	:param mb_sizes: List of micro-batch sizes
	:param time_limit: ILP solver time limit in seconds
	:return: solution dict with ``placement``, ``order``, and remat options per (stage, mb)
	:raises ImportError: if ``pulp`` is not installed
	:raises RuntimeError: if the ILP is infeasible
	"""
	try:
		import pulp as pl
	except ImportError:
		raise ImportError(
			"The 'pulp' package is required for ILP-based memory optimization. "
			"Install it with: pip install pulp"
		)

	mb_size = int(
		np.median(mb_sizes).item()
	)  # ILP does not support different micro-batch sizes anyway ; use the median as an approximation
	params = build_ilp_params(
		statistics, placement, scheduler, n_micro_batches, memory_budget, mb_size
	)

	p = params["p"]
	m = params["m"]
	nstages = params["nstages"]
	MGpu = params["MGpu"]
	sched = params["sched"]
	stages = params["stages"]
	plcmt = params["placement"]

	stages_ids = {rank: [i for i, k in enumerate(plcmt) if k == rank] for rank in range(p)}

	prob = pl.LpProblem("StageRemat", pl.LpMinimize)

	# ── Variables ──

	fwd_remat = [[{} for _ in range(m)] for _ in range(nstages)]
	bwd_remat = [[{} for _ in range(m)] for _ in range(nstages)]

	for i, stage in enumerate(stages):
		for j in range(m):
			for option in stage["fwd_remat_options"]:
				fwd_remat[i][j][option.name] = pl.LpVariable(f"{option.name}_{i}_{j}", cat=pl.LpBinary)
			for option in stage["bwd_remat_options"]:
				bwd_remat[i][j][option.name] = pl.LpVariable(f"{option.name}_{i}_{j}", cat=pl.LpBinary)

	# ── Helpers ──

	def time_op(op, i, j):
		T = stages[i]["T"]
		match op:
			case "f":
				return T["f"]
			case "b":
				return T["b"] + pl.lpSum(
					o.time_overhead * fwd_remat[i][j][o.name] for o in stages[i]["fwd_remat_options"]
				)
			case "w":
				return T["w"] + pl.lpSum(
					o.time_overhead * bwd_remat[i][j][o.name] for o in stages[i]["bwd_remat_options"]
				)

	def mem_kept(op, i, j):
		M = stages[i]["M"]
		match op:
			case "f":
				return M["f"] - pl.lpSum(
					o.mem_freed * fwd_remat[i][j][o.name] for o in stages[i]["fwd_remat_options"]
				)
			case "b":
				return M["b"] - pl.lpSum(
					o.mem_freed * bwd_remat[i][j][o.name] for o in stages[i]["bwd_remat_options"]
				)
			case "w":
				return M["w"]

	def delta_mem(op, i, j):
		match op:
			case "f":
				return mem_kept("f", i, j)
			case "b":
				return mem_kept("b", i, j) - mem_kept("f", i, j)
			case "w":
				return mem_kept("w", i, j) - mem_kept("b", i, j)

	def peak_mem(op, i, j):
		Mpeak = stages[i]["Mpeak"]
		peakF, peakB, peakW = Mpeak["f"], Mpeak["b"], Mpeak["w"]
		peakNoGrad = Mpeak["f_no_grad"]
		mf = stages[i]["M"]["f"]
		mb = stages[i]["M"]["b"]

		if op == "f":
			return peakF + fwd_remat[i][j]["full_fwd"] * (peakNoGrad - peakF)
		elif op == "b":
			recompute_peak = max(peakF, mf + peakB)
			return peakB + pl.lpSum(fwd_remat[i][j][o.name] for o in stages[i]["fwd_remat_options"]) * (
				recompute_peak - peakB
			)
		elif op == "w":
			recompute_peak = max(peakF, mf + peakB, mb + peakW)
			pk = peakW
			if "full_bwd" in bwd_remat[i][j]:
				pk += bwd_remat[i][j]["full_bwd"] * (recompute_peak - peakW)
			if "activations_bwd" in bwd_remat[i][j]:
				pk += bwd_remat[i][j]["activations_bwd"] * (peakNoGrad - peakW)
			return pk

	def mparams(rank):
		return 2 * sum(stages[s]["Mparams"] for s in stages_ids[rank])

	# ── Timing variables ──

	lower_bounds = [
		sum(sum(stages[s]["T"][op] for op in "fbw") * m for s in stages_ids[rank]) for rank in range(p)
	]

	E = [
		[
			[pl.LpVariable(f"E_{s}_{j}_{k}", cat=pl.LpContinuous, lowBound=0) for k in range(3)]
			for j in range(m)
		]
		for s in range(nstages)
	]

	T_rank = [None] * p
	for rank in range(p):
		first_op, first_s, first_mb = sched[rank][0]
		last_op, last_s, last_mb = sched[rank][-1]
		T_rank[rank] = (
			E[last_s][last_mb][_op_name_to_index(last_op)]
			- E[first_s][first_mb][_op_name_to_index(first_op)]
			+ time_op(first_op, first_s, first_mb)
		)

	Tmax = pl.LpVariable("Tmax", cat=pl.LpContinuous, lowBound=max(lower_bounds))

	# ── Objective ──

	prob += Tmax

	# ── Constraints ──

	for rank in range(p):
		prob += Tmax >= T_rank[rank]

	prob += E[0][0][0] - time_op("f", 0, 0) == 0

	for rank in range(p):
		n_ops = 3 * m * len(stages_ids[rank])
		memory_kept = [mparams(rank)]
		for k in range(n_ops):
			op, stage, j = sched[rank][k]

			if op == "f" and stage != 0:
				idx = _op_name_to_index("f")
				tcomm = stages[stage - 1]["Tcomm"] if plcmt[stage - 1] != rank else 0
				prob += E[stage][j][idx] >= E[stage - 1][j][idx] + time_op(op, stage, j) + tcomm
			elif op == "b" and stage != nstages - 1:
				idx = _op_name_to_index("b")
				tcomm = stages[stage]["Tcomm"] if plcmt[stage + 1] != rank else 0
				prob += E[stage][j][idx] >= E[stage + 1][j][idx] + time_op(op, stage, j) + tcomm

			if k != 0:
				op_prev, s_prev, j_prev = sched[rank][k - 1]
				prob += E[stage][j][_op_name_to_index(op)] >= E[s_prev][j_prev][
					_op_name_to_index(op_prev)
				] + time_op(op, stage, j)

			prob += pl.lpSum(memory_kept) + peak_mem(op, stage, j) <= MGpu
			memory_kept.append(delta_mem(op, stage, j))

	for stage in range(nstages):
		for j in range(m):
			prob += pl.lpSum(fwd_remat[stage][j][n] for n in fwd_remat[stage][j]) <= 1
			prob += pl.lpSum(bwd_remat[stage][j][n] for n in bwd_remat[stage][j]) <= 1

	# ── Solve ──

	_SOLVERS = ["CPLEX_PY", "CPLEX_CMD", "PULP_CBC_CMD"]
	for solver_name in _SOLVERS:
		solver = getattr(pl, solver_name)
		if solver().available():
			break
	else:
		raise RuntimeError(f"No solver found in {_SOLVERS}")

	logger.info(
		f"Solving ILP with {solver_name} solver, {time_limit} seconds time limit and {memory_budget} MB memory budget"
	)
	start = time.time()
	prob.solve(solver(msg=False, timeLimit=time_limit)) # TODO: add env variable to control solver/time limit
	end = time.time()
	solve_time = end - start

	if prob.status == pl.constants.LpStatusInfeasible:
		raise RuntimeError(
			f"ILP is infeasible: cannot fit the schedule within {memory_budget} MB. "
			"Try increasing memory_budget, reducing n_micro_batches or using a different scheduler."
		)

	# ── Extract solution ──

	strategy = {}
	for i, stage in enumerate(stages):
		for option in stage["fwd_remat_options"]:
			strategy.setdefault(option.name, []).append(
				[fwd_remat[i][j][option.name].value() for j in range(m)]
			)
		for option in stage["bwd_remat_options"]:
			strategy.setdefault(option.name, []).append(
				[bwd_remat[i][j][option.name].value() for j in range(m)]
			)

	objective = prob.objective.value()

	# Peak memory per rank (post-solve evaluation)
	peak_mems = []
	for rank in range(p):
		acc = mparams(rank)
		pk = 0
		for k_op in range(len(sched[rank])):
			op, s, j = sched[rank][k_op]
			pk = max(pk, acc + pl.value(peak_mem(op, s, j)))
			acc += pl.value(delta_mem(op, s, j))
		peak_mems.append(pk)

	order = [op for rank_sched in sched for op in rank_sched]

	solution = {
		"placement": list(plcmt),
		"objective": objective,
		"peak_mems": peak_mems,
		"order": order,
	}
	for option_name, values in strategy.items():
		solution[option_name] = values

	logger.info(
		f"ILP solved in {solve_time:.2f} seconds: objective={objective:.1f}ms, peak_mems={[f'{m:.0f}MB' for m in peak_mems]}"
	)
	return solution


# ── Schedule integration ─────────────────────────────────────────────────────


class RematScheduler:
	"""Wraps a base scheduler and applies ILP-solved remat decisions.

	For each ``FORWARD`` operation, if the solution says to recompute, adds a
	``REMAT_STRATEGY`` option that checkpoints the root module.

	For each ``BACKWARD_INPUTS`` operation, if the solution says to recompute
	activations or the full backward, inserts ``RECOMPUTE_FORWARD`` (and
	optionally ``RECOMPUTE_BACKWARD_INPUTS``) before the ``BACKWARD_PARAMS``.
	"""

	def __init__(self, base_scheduler, solution: dict):
		self.base_scheduler = base_scheduler
		self.solution = solution

	def __call__(self, placement, nmb, signatures):
		schedule = self.base_scheduler(placement, nmb, signatures)

		for op in schedule.copy():
			if op.op == OperationType.FORWARD:
				self._handle_forward_remat(op)
			elif op.op == OperationType.BACKWARD_INPUTS:
				self._handle_backward_remat(op, schedule)

		return schedule

	def _find_operation(self, sched, optype, block_id, mb_id):
		for i, op in enumerate(sched):
			if op.mb_id == mb_id and op.op == optype and op.block_id == block_id:
				return i
		return None

	def _handle_forward_remat(self, op):
		full_fwd = self.solution.get("full_fwd", [])
		if not full_fwd or full_fwd[op.block_id][op.mb_id] == 0:
			return

		def checkpoint_strategy(name, module):
			return name == ""

		op.options[OpOptions.REMAT_STRATEGY] = checkpoint_strategy

	def _handle_backward_remat(self, op, sched):
		activations_bwd = self.solution.get("activations_bwd", [])
		full_bwd = self.solution.get("full_bwd", [])

		recompute_activations = bool(activations_bwd and activations_bwd[op.block_id][op.mb_id])
		recompute_all = bool(full_bwd and full_bwd[op.block_id][op.mb_id])

		if not recompute_activations and not recompute_all:
			return

		op.options[OpOptions.RECOMPUTE_ACTIVATIONS] = True

		remat_fwd = Operation(op.block_id, op.mb_id, OperationType.RECOMPUTE_FORWARD, op.rank)
		w_idx = self._find_operation(sched, OperationType.BACKWARD_PARAMS, op.block_id, op.mb_id)
		if w_idx is not None:
			sched.insert(w_idx, remat_fwd)

		if not recompute_all:
			return

		remat_fwd.options[OpOptions.SAVE] = True
		op.options[OpOptions.RECOMPUTE_GRADIENTS] = True

		remat_bwd = Operation(op.block_id, op.mb_id, OperationType.RECOMPUTE_BACKWARD_INPUTS, op.rank)
		w_idx = self._find_operation(sched, OperationType.BACKWARD_PARAMS, op.block_id, op.mb_id)
		if w_idx is not None:
			sched.insert(w_idx, remat_bwd)
