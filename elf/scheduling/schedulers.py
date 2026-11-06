"""
Implementation of different state-of-the-art scheduling algorithms.
A schedule is a list of operations (see :class:`~elf.scheduling.scheduling.Operation`) that will be executed in order by each device.
Every rank should generate the entire schedule for all ranks, in order to detect and fix cycles/deadlocks.
"""

import logging

from elf.registry import Scheduler
from elf.scheduling.scheduling import OpOptions, OperationType, Operation, RematMode

logger = logging.getLogger(__name__)

__all__ = [
	"FixedSchedule",
	"generate_afab_schedule",
	"generate_1f1b_schedule",
	"generate_hanayo_schedule",
	"generate_full_remat_schedule",
	"generate_zbh1_schedule",
	"generate_zbh2_schedule",
	"generate_zbv_schedule",
	"generate_inference_schedule",
]


def _remat_root(name, _):
	"""The ``REMAT_STRATEGY`` that checkpoints a whole stage."""
	return RematMode.CHECKPOINT if name == "" else None


def _add_forward_pass(schedule, placement, block_id, mb_id, rank, options=None):
	options = options or {}
	schedule.append(Operation(block_id, mb_id, OperationType.FORWARD, rank, options=options))
	if block_id == len(placement) - 1:
		schedule.append(Operation(block_id, mb_id, OperationType.LOSS_FORWARD, rank))


def _add_backward_pass(schedule, placement, block_id, mb_id, rank, options=None):
	options = options or {}
	if block_id == len(placement) - 1:
		schedule.append(Operation(block_id, mb_id, OperationType.LOSS_BACKWARD, rank))
	schedule.append(Operation(block_id, mb_id, OperationType.BACKWARD_INPUTS, rank, options=options))


def _add_backward_params(schedule, block_id, mb_id, rank, options=None):
	options = options or {}
	schedule.append(Operation(block_id, mb_id, OperationType.BACKWARD_PARAMS, rank, options=options))


def generate_afab_schedule(placement, n_micro_batches, profiles):
	schedule = []
	ranks = placement.ranks

	for rank in ranks:
		ids = [i for i in range(len(placement)) if placement[i] == rank]
		# All forward
		for i in range(n_micro_batches):
			for id_ in ids:
				_add_forward_pass(schedule, placement, id_, i, rank)
		# All backward
		for i in range(n_micro_batches):
			for id_ in reversed(ids):
				_add_backward_pass(schedule, placement, id_, i, rank)
				_add_backward_params(schedule, id_, i, rank)

	return schedule


def generate_1f1b_schedule(placement, n_micro_batches, profiles):
	schedule = []
	ranks = placement.ranks
	n_ranks = len(ranks)
	n_stages = len(placement)
	stages_per_rank = n_stages // n_ranks

	for idx, rank in enumerate(ranks):
		ids = [i for i in range(n_stages) if placement[i] == rank]
		fwds = [0 for _ in range(stages_per_rank)]
		bwds = [0 for _ in range(stages_per_rank)]
		curr_f = 0
		curr_b = stages_per_rank - 1

		def f():
			# Perform the next F
			# Increasing microbatch order, and increasing stage every n_ranks F
			nonlocal curr_f
			if fwds[curr_f] < n_micro_batches:
				_add_forward_pass(schedule, placement, ids[curr_f], fwds[curr_f], rank)
				fwds[curr_f] += 1
			if fwds[curr_f] % n_ranks == 0 or fwds[curr_f] == n_micro_batches:
				curr_f = (curr_f + 1) % stages_per_rank

		def b():
			# Perform the next B
			# Increasing microbatch order, and decreasing stage every n_ranks B
			nonlocal curr_b
			if bwds[curr_b] < n_micro_batches:
				_add_backward_pass(schedule, placement, ids[curr_b], bwds[curr_b], rank)
				_add_backward_params(schedule, ids[curr_b], bwds[curr_b], rank)
				bwds[curr_b] += 1
			if bwds[curr_b] % n_ranks == 0 or bwds[curr_b] == n_micro_batches:
				curr_b = (curr_b - 1) % stages_per_rank

		n_warmups = min(n_micro_batches * stages_per_rank, n_ranks * stages_per_rank - (idx + 1))
		for _ in range(n_warmups):
			f()

		while (
			sum(fwds) < n_micro_batches * stages_per_rank or sum(bwds) < n_micro_batches * stages_per_rank
		):
			# Steady + cooldown state merged in one because F are skipped when already done
			f()
			b()

	return schedule


def generate_hanayo_schedule(placement, n_micro_batches, profiles):
	schedule = []
	ranks = placement.ranks
	n_ranks = len(ranks)
	n_stages = len(placement)
	n_waves = n_stages // (n_ranks * 2)

	ids = {rank: [i for i in range(n_stages) if placement[i] == rank] for rank in ranks}

	done = {rank: 0 for rank in ranks}  # mb computed forward :)
	enod = {rank: 0 for rank in ranks}  # mb computed backward (:

	# Warmup
	for idx, rank in enumerate(ranks):
		done[rank] = min(n_ranks - idx, n_micro_batches)
		for i in range(done[rank]):
			id_ = ids[rank][0]
			_add_forward_pass(schedule, placement, id_, i, rank)

	for w in range(n_waves):
		for mb in range(n_micro_batches):
			for rank in reversed(ranks):
				id_ = ids[rank][2 * w + 1]
				_add_forward_pass(schedule, placement, id_, mb, rank)

				if done[rank] < n_micro_batches * n_waves:
					id_ = ids[rank][2 * (done[rank] // n_micro_batches)]
					_add_forward_pass(schedule, placement, id_, done[rank] % n_micro_batches, rank)
					done[rank] += 1
				else:
					id_ = ids[rank][-1 - 2 * (enod[rank] // n_micro_batches)]
					_add_backward_pass(schedule, placement, id_, enod[rank] % n_micro_batches, rank)
					_add_backward_params(schedule, id_, enod[rank] % n_micro_batches, rank)
					enod[rank] += 1

	for w in range(n_waves):
		for mb in range(n_micro_batches):
			for rank in reversed(ranks):
				id_ = ids[rank][-2 * (w + 1)]
				_add_backward_pass(schedule, placement, id_, mb, rank)
				_add_backward_params(schedule, id_, mb, rank)

				if enod[rank] < n_micro_batches * n_waves:
					id_ = ids[rank][-1 - 2 * (enod[rank] // n_micro_batches)]
					_add_backward_pass(schedule, placement, id_, enod[rank] % n_micro_batches, rank)
					_add_backward_params(schedule, id_, enod[rank] % n_micro_batches, rank)
					enod[rank] += 1

	return schedule


def generate_full_remat_schedule(placement, n_micro_batches, profiles):
	schedule = []
	ranks = placement.ranks

	# All ranks do afab
	for rank in ranks[:-1]:
		ids = [i for i in range(len(placement)) if placement[i] == rank]
		# All forward
		for i in range(n_micro_batches):
			for id_ in ids:
				_add_forward_pass(
					schedule, placement, id_, i, rank, options={OpOptions.REMAT_STRATEGY: _remat_root}
				)
		# All backward
		for i in range(n_micro_batches):
			for id_ in reversed(ids):
				schedule.append(
					Operation(id_, i, OperationType.RECOMPUTE_FORWARD, rank)
				)  # recomputation happens here
				_add_backward_pass(schedule, placement, id_, i, rank)
				_add_backward_params(schedule, id_, i, rank)

	# Last rank does 1f1b
	rank = ranks[-1]
	ids = [i for i in range(len(placement)) if placement[i] == rank]
	for i in range(n_micro_batches):
		for id_ in ids:
			_add_forward_pass(schedule, placement, id_, i, rank)

		for id_ in reversed(ids):
			_add_backward_pass(schedule, placement, id_, i, rank)
			_add_backward_params(schedule, id_, i, rank)

	return schedule


def generate_zbh1_schedule(placement, n_micro_batches, profiles):
	schedule = []
	ranks = placement.ranks
	n_ranks = len(ranks)

	for block_id, rank in enumerate(placement):
		n_warmups = min(n_ranks - block_id, n_micro_batches)
		f = 0
		b = 0
		w = 0

		for _ in range(n_warmups):
			_add_forward_pass(schedule, placement, block_id, f, rank)
			f += 1

		_add_backward_pass(schedule, placement, block_id, b, rank)
		b += 1

		for _ in range(block_id):
			if f < n_micro_batches:
				_add_forward_pass(schedule, placement, block_id, f, rank)
				f += 1
			if b < n_micro_batches:
				_add_backward_pass(schedule, placement, block_id, b, rank)
				b += 1

		while f < n_micro_batches or b < n_micro_batches or w < n_micro_batches:
			if w < n_micro_batches:
				_add_backward_params(schedule, block_id, w, rank)
				w += 1
			if f < n_micro_batches:
				_add_forward_pass(schedule, placement, block_id, f, rank)
				f += 1
			if b < n_micro_batches:
				_add_backward_pass(schedule, placement, block_id, b, rank)
				b += 1

	return schedule


def generate_zbh2_schedule(placement, n_micro_batches, profiles):
	schedule = []
	ranks = placement.ranks
	n_ranks = len(ranks)

	for block_id, rank in enumerate(placement):
		# We don't support interleaving yet, only consider one block per rank
		# ids = [i for i in range(len(placement)) if placement[i] == rank]
		n_warmups = min(n_micro_batches, 2 * (n_ranks - block_id) - 1)
		f = 0
		b = 0
		w = 0
		for _ in range(n_warmups):
			_add_forward_pass(schedule, placement, block_id, f, rank)
			f += 1

		_add_backward_pass(schedule, placement, block_id, b, rank)
		b += 1

		for i in range(n_micro_batches - 1 - n_warmups):
			_add_forward_pass(schedule, placement, block_id, f, rank)
			f += 1

			_add_backward_pass(schedule, placement, block_id, b, rank)
			b += 1

		while f < n_micro_batches or b < n_micro_batches or w < n_micro_batches:
			if w < n_micro_batches:
				_add_backward_params(schedule, block_id, w, rank)
				w += 1
			if f < n_micro_batches:
				_add_forward_pass(schedule, placement, block_id, f, rank)
				f += 1
			if b < n_micro_batches:
				_add_backward_pass(schedule, placement, block_id, b, rank)
				b += 1

	return schedule


def generate_zbv_schedule(placement, n_micro_batches, profiles):
	"""
	https://arxiv.org/abs/2401.10241 Section 6 paragraph 3
	"""
	schedule = []
	ranks = placement.ranks
	n_ranks = len(ranks)
	stages_per_device = len(placement) // n_ranks

	for idx, rank in enumerate(ranks):
		ids = [i for i in range(len(placement)) if placement[i] == rank]

		n_f = [0 for _ in range(stages_per_device)]
		n_b = [0 for _ in range(stages_per_device)]
		n_w = [0 for _ in range(stages_per_device)]

		def local_add_forward_pass(schedule, placement, k, rank):
			if n_f[k] < n_micro_batches:
				_add_forward_pass(schedule, placement, ids[k], n_f[k], rank)
				n_f[k] += 1

		def local_add_backward_pass(schedule, placement, k, rank):
			if n_b[k] < n_micro_batches:
				_add_backward_pass(schedule, placement, ids[k], n_b[k], rank)
				n_b[k] += 1

		def local_add_backward_params(schedule, k, rank):
			if n_w[k] < n_micro_batches:
				_add_backward_params(schedule, ids[k], n_w[k], rank)
				n_w[k] += 1

		# Phase 1: Warmup of first stage
		for _ in range(2 * (n_ranks - idx) - 1):
			local_add_forward_pass(schedule, placement, 0, rank)

		# Phase 2: Alternating warmup of all stages
		current = 1
		for _ in range(2 * idx):
			local_add_forward_pass(schedule, placement, current, rank)
			current = (current + 1) % (stages_per_device)

		# Phase 3: FBW of last stage
		for _ in range(n_ranks - idx):
			local_add_forward_pass(schedule, placement, -1, rank)
			local_add_backward_pass(schedule, placement, -1, rank)
			local_add_backward_params(schedule, -1, rank)

		# Phase 4: FBW of all stages (steady state)
		current = 0
		# do all f
		while sum(n_f) < n_micro_batches * len(n_f):
			local_add_forward_pass(schedule, placement, current, rank)
			local_add_backward_pass(schedule, placement, current, rank)
			local_add_backward_params(schedule, current, rank)
			current = (current + 1) % (stages_per_device)

		# Phase 5: Alternating B
		current = 0
		for _ in range(2 * idx + 1):
			local_add_backward_pass(schedule, placement, current, rank)
			current = (current + 1) % (stages_per_device)

		# Phase 6: Alternating B & W
		current = 0
		while sum(n_w) < n_micro_batches * len(n_w):
			current_w = current
			if n_w[current] == n_b[current]:
				current_w = (current_w + 1) % (stages_per_device)

			local_add_backward_params(schedule, current_w, rank)
			local_add_backward_pass(schedule, placement, current, rank)

	return schedule


def generate_inference_schedule(placement, n_micro_batches):
	schedule = []

	for id_ in range(len(placement)):
		for mb in range(n_micro_batches):
			schedule.append(Operation(id_, mb, OperationType.FORWARD, placement[id_]))

	return schedule


class FixedSchedule(Scheduler):
	"""
	Scheduler that uses a dictionary representation of the schedule.

	The expected format is the entire schedule as:
	{
		"order": [
			(optype, block_id, mb_id),
			(optype, block_id, mb_id),
			...
		]
	}
	"""

	def __init__(self, schedule_dict):
		super().__init__(self._schedule, self._check)
		self.schedule_dict = schedule_dict

	def _schedule(self, placement, n_micro_batches, profiles):
		schedule = []
		order = self.schedule_dict["order"]

		for optype, block_id, mb_id, *opts in order:
			block_id = int(block_id)
			mb_id = int(mb_id)
			options = {}

			match optype:
				case "f":
					if "remat" in opts:
						options[OpOptions.REMAT_STRATEGY] = _remat_root

					_add_forward_pass(schedule, placement, block_id, mb_id, placement[block_id], options)
				case "b":
					if "rematf" in opts:
						options[OpOptions.RECOMPUTE_ACTIVATIONS] = True
					if "rematb" in opts:
						options[OpOptions.RECOMPUTE_ACTIVATIONS] = True
						options[OpOptions.RECOMPUTE_GRADIENTS] = True

					_add_backward_pass(schedule, placement, block_id, mb_id, placement[block_id], options)
				case "w":
					_add_backward_params(schedule, block_id, mb_id, placement[block_id], options)
				case "rf":
					schedule.append(
						Operation(block_id, mb_id, OperationType.RECOMPUTE_FORWARD, placement[block_id])
					)
				case "rbf":
					schedule.append(
						Operation(
							block_id,
							mb_id,
							OperationType.RECOMPUTE_FORWARD_FOR_PARAMS,
							placement[block_id],
							options=options,
						)
					)
				case "rbb":
					schedule.append(
						Operation(block_id, mb_id, OperationType.RECOMPUTE_BACKWARD_INPUTS, placement[block_id])
					)

		return schedule

	def _check(self, placement, nmb):
		"""
		Check that the stored schedule was built for ``len(placement)`` blocks and ``nmb`` micro
		batches: every operation references an existing block and micro batch, and each
		(block, micro batch) pair has exactly one forward pass.
		"""
		forwards = []
		for optype, block_id, mb_id, *_ in self.schedule_dict["order"]:
			block_id, mb_id = int(block_id), int(mb_id)
			if not (0 <= block_id < len(placement) and 0 <= mb_id < nmb):
				return False
			if optype == "f":
				forwards.append((block_id, mb_id))

		return len(forwards) == len(placement) * nmb and len(set(forwards)) == len(forwards)
