"""ILP-based rematerialization optimization for pipeline schedules.

Profiles pipeline stages, solves an integer linear program to decide which
operations to recompute, and wraps the scheduler to apply those decisions.

Usage from Pipeline:
	Set ``memory_budget`` (MB) in PipelineConfig.  The Pipeline profiles the stages at setup
	(:func:`elf.profiling.profile_pipeline`) and solves the ILP lazily on the first training step,
	once the number of micro-batches is known.

For more details, see: Adrien Aguila--Multner, Olivier Beaumont, Lionel Eyraud-Dubois, Julia Gusak. Optimized Forward-Backward Rematerialization for Memory-Efficient Pipeline Parallel Training. 2025. https://inria.hal.science/hal-05151601
"""

import os
import time
import pickle
import logging
import threading

from dataclasses import dataclass

from elf.topology import Placement
import torch.distributed as dist

from elf.profiling import StageProfile
from elf.registry import resolve, Scheduler, SCHEDULERS
from elf.scheduling import OpOptions, OperationType, Operation, RematMode

logger = logging.getLogger(__name__)


# ── ILP solver ───────────────────────────────────────────────────────────────

#: Rematerialization options of the forward pass, at most one per (stage, micro-batch).
FORWARD_REMAT_OPTIONS = ("full_fwd",)
#: Rematerialization options of the backward pass, at most one per (stage, micro-batch).
BACKWARD_REMAT_OPTIONS = ("activations_bwd", "full_bwd")

F, B, W = OperationType.FORWARD, OperationType.BACKWARD_INPUTS, OperationType.BACKWARD_PARAMS


@dataclass
class RematSolution:
	placement: list[int]
	order: list[Operation]  # schedule
	# decision variable matrices
	forward_remat: dict[str, list[list[int]]]  # [option name][stage][microbatch]
	backward_remat: dict[str, list[list[int]]]  # [option name][stage][microbatch]
	objective: float  # objective value
	peak_mems: list[float]  # peak memory per rank


def solve_remat(
	profiles: list[StageProfile],
	placement: list[int],
	scheduler: Scheduler,
	n_micro_batches: int,
	memory_budgets: list[int],
) -> RematSolution:
	"""Solve the rematerialization ILP.

	Given a profile for each stage, finds the per-(stage, microbatch)
	remat strategy that minimizes iteration time while keeping peak memory
	within the given budget.

	:param profiles: one profile per stage, from :func:`elf.profiling.profile_pipeline`
	:type profiles: list[StageProfile]
	:param placement: stage-to-rank mapping
	:type placement: list[int]
	:param scheduler: resolved scheduler callable
	:type scheduler: Scheduler
	:param n_micro_batches: number of micro-batches
	:type n_micro_batches: int
	:param memory_budgets: GPU memory budgets in MB, one per rank
	:type memory_budgets: list[int]
	:return: the schedule's placement and order, the chosen remat options per (stage, mb), the
		makespan and the peak memory of each rank
	:rtype: RematSolution
	:raises ImportError: if ``pulp`` is not installed
	:raises ValueError: if there is not exactly one profile per stage
	:raises RuntimeError: if the ILP is infeasible
	"""
	try:
		import pulp as pl
	except ImportError:
		raise ImportError(
			"The 'pulp' package is required for ILP-based memory optimization. "
			"Install it with: pip install pulp"
		)

	p = len(set(placement))
	m = n_micro_batches
	nstages = len(placement)
	assert len(profiles) == nstages, (
		f"Expected one profile per stage ({nstages}), got {len(profiles)}"
	)
	assert len(memory_budgets) == p, (
		f"Expected one memory budget per rank ({p}), got {len(memory_budgets)}"
	)

	sched = [[] for _ in range(p)]
	for op in scheduler(placement, m, profiles):
		if op.op in (F, B, W):
			sched[op.rank].append(op)

	stages_ids = {rank: [i for i, k in enumerate(placement) if k == rank] for rank in range(p)}

	prob = pl.LpProblem("StageRemat", pl.LpMinimize)

	# ── Variables ──

	fwd_remat = [
		[
			{o: pl.LpVariable(f"{o}_{i}_{j}", cat=pl.LpBinary) for o in FORWARD_REMAT_OPTIONS}
			for j in range(m)
		]
		for i in range(nstages)
	]
	bwd_remat = [
		[
			{o: pl.LpVariable(f"{o}_{i}_{j}", cat=pl.LpBinary) for o in BACKWARD_REMAT_OPTIONS}
			for j in range(m)
		]
		for i in range(nstages)
	]

	# ── Helpers ──

	def time_op(op, i, j):
		profile = profiles[i]
		match op:
			case OperationType.FORWARD:
				return profile.t_forward
			case OperationType.BACKWARD_INPUTS:
				return profile.t_backward_inputs + fwd_remat[i][j]["full_fwd"] * profile.t_forward
			case OperationType.BACKWARD_PARAMS:
				return (
					profile.t_backward_params
					+ bwd_remat[i][j]["activations_bwd"] * profile.t_forward
					+ bwd_remat[i][j]["full_bwd"] * (profile.t_forward + profile.t_backward_inputs)
				)

	def mem_kept(op, i, j):
		profile = profiles[i]
		held = profile.m_inputs + profile.m_outputs
		match op:
			case OperationType.FORWARD:
				return profile.m_forward - fwd_remat[i][j]["full_fwd"] * (profile.m_forward - held)
			case OperationType.BACKWARD_INPUTS:
				# a stage that replays its forward before W keeps its inputs until then to start
				# the replay from, whether or not its graph would have kept them
				return (
					profile.m_backward_inputs
					- bwd_remat[i][j]["activations_bwd"] * (profile.m_saved_inputs - profile.m_inputs)
					- bwd_remat[i][j]["full_bwd"] * (profile.m_backward_inputs - held)
				)
			case OperationType.BACKWARD_PARAMS:
				return profile.m_backward_params

	def delta_mem(op, i, j):
		match op:
			case OperationType.FORWARD:
				return mem_kept(F, i, j)
			case OperationType.BACKWARD_INPUTS:
				return mem_kept(B, i, j) - mem_kept(F, i, j)
			case OperationType.BACKWARD_PARAMS:
				return mem_kept(W, i, j) - mem_kept(B, i, j)

	def peak_mem(op, i, j):
		profile = profiles[i]
		peakF = profile.m_peak_forward
		peakB = profile.m_peak_backward_inputs
		peakW = profile.m_peak_backward_params
		peakNoGrad = profile.m_peak_forward_no_grad
		mf = profile.m_forward
		mb = profile.m_backward_inputs

		match op:
			case OperationType.FORWARD:
				# If we full recompute, it is equivalent to running under no_grad
				return peakF + fwd_remat[i][j]["full_fwd"] * (peakNoGrad - peakF)
			case OperationType.BACKWARD_INPUTS:
				# Conservative approximation: if anything is recomputed, consider that the peak can be as high as a full F + B
				recompute_peak = max(peakF, mf + peakB)
				return peakB + fwd_remat[i][j]["full_fwd"] * (recompute_peak - peakB)
			case OperationType.BACKWARD_PARAMS:
				recompute_peak = max(peakF, mf + peakB, mb + peakW)
				# peak of FBW: can be during F, during B with MF stored, or during W with MB stored
				pk = peakW + bwd_remat[i][j]["full_bwd"] * (recompute_peak - peakW)
				# if we only recompute activations, the F is ran under no_grad
				# during recomputation: peakNoGrad
				# during W: mba + peakW
				pk += bwd_remat[i][j]["activations_bwd"] * (
					max(peakNoGrad, profile.m_saved_inputs + peakW) - peakW
				)
				return pk

	def mparams(rank):
		return 2 * sum(profiles[s].m_parameters for s in stages_ids[rank])  # params + gradients

	# ── Timing variables ──

	lower_bounds = [
		m
		* sum(
			profiles[s].t_forward + profiles[s].t_backward_inputs + profiles[s].t_backward_params
			for s in stages_ids[rank]
		)
		for rank in range(p)
	]

	E = [
		[
			{t: pl.LpVariable(f"E_{s}_{j}_{t.name}", cat=pl.LpContinuous, lowBound=0) for t in (F, B, W)}
			for j in range(m)
		]
		for s in range(nstages)
	]

	def end(op):
		return E[op.block_id][op.mb_id][op.op]

	T_rank = [None] * p
	for rank in range(p):
		first, last = sched[rank][0], sched[rank][-1]
		T_rank[rank] = end(last) - end(first) + time_op(first.op, first.block_id, first.mb_id)

	Tmax = pl.LpVariable("Tmax", cat=pl.LpContinuous, lowBound=max(lower_bounds))

	# ── Objective ──

	prob += Tmax

	# ── Constraints ──

	for rank in range(p):
		prob += Tmax >= T_rank[rank]

	prob += E[0][0][F] - time_op(F, 0, 0) == 0

	for rank in range(p):
		memory_kept = [mparams(rank)]
		for k, op in enumerate(sched[rank]):
			stage, j = op.block_id, op.mb_id

			if op.op == F and stage != 0:
				tcomm = profiles[stage - 1].t_transfer if placement[stage - 1] != rank else 0
				prob += E[stage][j][F] >= E[stage - 1][j][F] + time_op(F, stage, j) + tcomm
			elif op.op == B and stage != nstages - 1:
				tcomm = profiles[stage].t_transfer if placement[stage + 1] != rank else 0
				prob += E[stage][j][B] >= E[stage + 1][j][B] + time_op(B, stage, j) + tcomm

			if k != 0:
				prob += end(op) >= end(sched[rank][k - 1]) + time_op(op.op, stage, j)

			prob += pl.lpSum(memory_kept) + peak_mem(op.op, stage, j) <= memory_budgets[rank]
			memory_kept.append(delta_mem(op.op, stage, j))

	for stage in range(nstages):
		for j in range(m):
			prob += pl.lpSum(fwd_remat[stage][j].values()) <= 1
			prob += pl.lpSum(bwd_remat[stage][j].values()) <= 1

	# ── Solve ──
	solver_name = os.getenv("ELF_ILP_SOLVER", None)
	if solver_name is not None:
		assert solver_name in pl.listSolvers(onlyAvailable=True), f"Solver {solver_name} not found"
		solver = getattr(pl, solver_name)
		if not solver().available():
			raise RuntimeError(
				f"ILP solver set to {solver_name} but it is not available on this machine."
			)
	else:
		_SOLVERS = ["GUROBI", "CPLEX_PY", "CPLEX_CMD", "PULP_CBC_CMD"]
		for solver_name in _SOLVERS:
			solver = getattr(pl, solver_name)
			if solver().available():
				break
		else:
			raise RuntimeError(f"No available solver found in {_SOLVERS}")

	time_limit = int(os.getenv("ELF_ILP_TIME_LIMIT", 60))
	gap_rel_limit = float(os.getenv("ELF_ILP_GAP_REL_LIMIT", 0.03))

	logger.info(
		f"Solving ILP with {solver_name} solver, {SCHEDULERS.get_key(scheduler)} scheduler, {time_limit} seconds time limit and {memory_budgets} MB memory budget"
	)
	start = time.time()
	prob.solve(solver(msg=False, timeLimit=time_limit, gapRel=gap_rel_limit))
	end = time.time()
	solve_time = end - start

	if prob.status == pl.constants.LpStatusInfeasible:
		raise RuntimeError(
			f"ILP is infeasible: cannot fit the schedule within {memory_budgets} MB. "
			"Try increasing memory_budget, reducing n_micro_batches or using a different scheduler."
		)

	# ── Extract solution ──
	forward_remat = {
		o: [[fwd_remat[i][j][o].value() for j in range(m)] for i in range(nstages)]
		for o in FORWARD_REMAT_OPTIONS
	}
	backward_remat = {
		o: [[bwd_remat[i][j][o].value() for j in range(m)] for i in range(nstages)]
		for o in BACKWARD_REMAT_OPTIONS
	}

	objective = prob.objective.value()
	order = [op for rank_sched in sched for op in rank_sched]

	# Peak memory per rank (post-solve evaluation)
	peak_mems = []
	for rank in range(p):
		acc = mparams(rank)
		pk = 0
		for op in sched[rank]:
			pk = max(pk, acc + pl.value(peak_mem(op.op, op.block_id, op.mb_id)))
			acc += pl.value(delta_mem(op.op, op.block_id, op.mb_id))
		peak_mems.append(pk)

	solution = RematSolution(
		placement=list(placement),
		order=order,
		forward_remat=forward_remat,
		backward_remat=backward_remat,
		objective=objective,
		peak_mems=peak_mems,
	)
	logger.info(
		f"ILP solved in {solve_time:.2f} seconds: objective={objective:.1f}ms, peak_mems={[f'{m / 1024:.2f}GiB' for m in peak_mems]}"
	)
	return solution


# ── Schedule integration ─────────────────────────────────────────────────────


class RematScheduler:
	"""Wraps a base scheduler and applies ILP-solved remat decisions.

	For each ``FORWARD`` operation, if the solution says to recompute, adds a
	``REMAT_STRATEGY`` option that checkpoints the root module.

	For each ``BACKWARD_INPUTS`` operation, if the solution says to recompute
	activations or the full backward, inserts ``RECOMPUTE_FORWARD_FOR_PARAMS`` (and
	optionally ``RECOMPUTE_BACKWARD_INPUTS``) before the ``BACKWARD_PARAMS``.

	.. warning::
		This scheduler only supports the base remat options: full_fwd, activations_bwd and full_bwd.
		Adding user-defined remat options is not yet possible.
	"""

	def __init__(self, base_scheduler, solution: RematSolution):
		self.base_scheduler = base_scheduler
		self.solution = solution

	@property
	def makespan(self):
		return self.solution.objective

	def __call__(self, placement, nmb, profiles):
		schedule = self.base_scheduler(placement, nmb, profiles)

		for op in schedule.copy():
			if op.op == OperationType.FORWARD:
				self._handle_forward_remat(op, schedule)
			elif op.op == OperationType.BACKWARD_INPUTS:
				self._handle_backward_remat(op, schedule)

		return schedule

	def _find_operation(self, sched, optype, block_id, mb_id):
		for i, op in enumerate(sched):
			if op.mb_id == mb_id and op.op == optype and op.block_id == block_id:
				return i
		return None

	def _handle_forward_remat(self, op, sched):
		full_fwd = self.solution.forward_remat.get("full_fwd", [])
		if not full_fwd or full_fwd[op.block_id][op.mb_id] == 0:
			return

		def checkpoint_strategy(name, _):
			return RematMode.CHECKPOINT if name == "" else None

		op.options[OpOptions.REMAT_STRATEGY] = checkpoint_strategy

		# Add the recomputation
		# If we don't, it is done implicitly during B, which means it cannot start before the end of RB
		rf = Operation(op.block_id, op.mb_id, OperationType.RECOMPUTE_FORWARD, op.rank)
		b_idx = self._find_operation(sched, OperationType.BACKWARD_INPUTS, op.block_id, op.mb_id)
		if b_idx is not None:
			sched.insert(b_idx, rf)

	def _handle_backward_remat(self, op, sched):
		activations_bwd = self.solution.backward_remat.get("activations_bwd", [])
		full_bwd = self.solution.backward_remat.get("full_bwd", [])

		recompute_activations = bool(activations_bwd and activations_bwd[op.block_id][op.mb_id])
		recompute_all = bool(full_bwd and full_bwd[op.block_id][op.mb_id])

		if not recompute_activations and not recompute_all:
			return

		op.options[OpOptions.RECOMPUTE_ACTIVATIONS] = True

		remat_fwd = Operation(
			op.block_id, op.mb_id, OperationType.RECOMPUTE_FORWARD_FOR_PARAMS, op.rank
		)
		w_idx = self._find_operation(sched, OperationType.BACKWARD_PARAMS, op.block_id, op.mb_id)
		if w_idx is not None:
			sched.insert(w_idx, remat_fwd)

		if not recompute_all:
			return

		op.options[OpOptions.RECOMPUTE_GRADIENTS] = True

		remat_bwd = Operation(op.block_id, op.mb_id, OperationType.RECOMPUTE_BACKWARD_INPUTS, op.rank)
		w_idx = self._find_operation(sched, OperationType.BACKWARD_PARAMS, op.block_id, op.mb_id)
		if w_idx is not None:
			sched.insert(w_idx, remat_bwd)


def _is_picklable(obj) -> bool:
	try:
		pickle.loads(pickle.dumps(obj))
	except Exception:
		return False
	return True


def solve_with_keepalive(timeout, worker, *args, **kwargs):
	"""Solve the ILP on *worker* and broadcast the outcome to every other rank.

	The worker always broadcasts exactly one terminal message per call, either a
	:class:`RematSolution` or an exception, so that a failed solve raises on every
	rank at the same iteration instead of leaving the other ranks blocked in
	``broadcast_object_list``.
	"""
	if dist.get_rank() == worker:
		# solve_remat, but sending keepalive signals to other ranks regularly during ILP solve (to avoid NCCL timeouts)
		stop_keepalive = threading.Event()

		def keepalive():
			while not stop_keepalive.wait(timeout=30.0):
				dist.broadcast_object_list(["keepalive"], src=worker)

		keepalive_thread = threading.Thread(target=keepalive, name="keepalive", daemon=True)
		# Keep sending keepalives while solve_remat() is running.
		keepalive_thread.start()

		try:
			result = solve_remat(*args, **kwargs)
		except Exception as e:
			result = e if _is_picklable(e) else RuntimeError(f"{type(e).__name__}: {e}")
		finally:
			# Make sure the keepalive thread has finished its collective
			# before the main thread starts another broadcast.
			stop_keepalive.set()
			keepalive_thread.join()

		dist.broadcast_object_list([result], src=worker)
		if isinstance(result, Exception):
			raise result
		return result
	else:
		while True:
			received = [None]
			dist.broadcast_object_list(received, src=worker)

			if received[0] == "keepalive":
				continue

			if isinstance(received[0], Exception):
				raise received[0]

			return received[0]


def stagerematify(scheduler: Scheduler, budgets: list[int], worker=0):  # noqa: F821 # type: ignore
	"""
	Transform a scheduler into a scheduler + ILP-solved remat strategy.

	:param scheduler: The base scheduler to transform.
	:type scheduler: Scheduler
	:param budget: The memory budget to use for the ILP in MB.
	:type budget: int
	:param worker: The worker rank to solve the ILP on.
	:type worker: int

	:return: A new scheduler, respecting the scheduler interface, that does both scheduling and remat optimization via ILP formulation.
	"""

	def StageRemat(placement: list[int], nmb: int, profiles: list[StageProfile]) -> list[Operation]:
		solution = solve_with_keepalive(30, worker, profiles, placement, scheduler, nmb, budgets)
		remat_scheduler = RematScheduler(scheduler, solution)
		schedule = remat_scheduler(placement, nmb, profiles)
		return schedule

	return Scheduler(StageRemat, scheduler.check)


class AutoScheduler(Scheduler):
	"""
	Scheduler that solves StageRemat for all base schedulers and uses the best objective value.

	It accepts every placement: a placement is valid as long as one of the registered schedulers
	accepts it.
	"""

	def __init__(self, budgets: list[int], worker=0):
		super().__init__(self._schedule)
		self.budgets = budgets
		self.worker = worker
		self.schedulers = SCHEDULERS.available()
		self.schedulers.remove("auto")  # auto is a dummy scheduler that we are currently overriding
		self.schedulers.remove(
			"fixed"
		)  # fixed is another special case that needs an external input (the schedule)
		self.schedulers.remove(
			"full_remat"
		)  # full_remat is not a base scheduler, remat is already included in it
		self.schedulers.remove("inference")  # we're not doing inference

	def _schedule(
		self, placement: Placement, nmb: int, profiles: list[StageProfile]
	) -> list[Operation]:
		best = (float("inf"), None, None)  # (objective, scheduler_name, solution)
		for scheduler_name in self.schedulers:
			scheduler = resolve(scheduler_name, SCHEDULERS)
			if not scheduler.check(
				placement, nmb
			):  # check is placement + nmb are valid for this scheduler, as defined in the registry
				continue

			try:
				solution = solve_with_keepalive(
					30, self.worker, profiles, placement, scheduler, nmb, self.budgets
				)
				objective = solution.objective
				if objective < best[0]:
					best = (objective, scheduler_name, solution)

			except Exception as e:
				if dist.get_rank() == self.worker:
					logger.info(f"Error solving ILP for {scheduler_name} scheduler: {e}. ")

		objective, scheduler_name, solution = best

		if scheduler_name is None:
			raise RuntimeError("No valid ILP solution found among all schedulers.")
		if dist.get_rank() == self.worker:
			logger.info(f"Best scheduler: {scheduler_name} with {objective:.2f}ms makespan")
		rscheduler = RematScheduler(resolve(scheduler_name, SCHEDULERS), solution)(
			placement, nmb, profiles
		)
		return rscheduler
