"""
Manage different types of schedule.
A schedule is a list of operations (see Operation) that will be executed in order by each device.
Every rank should generate the entire schedule for all ranks, in order to detect and fix cycles/deadlocks.
"""

from .task_graph import OperationType, Operation
import logging

logger = logging.getLogger("schedule")


def _add_forward_pass(schedule, block_id, mb_id, rank, options):
	for op_type in [OperationType.RECV_FORWARD, OperationType.FORWARD, OperationType.SEND_FORWARD]:
		schedule.append(Operation(block_id, mb_id, op_type, rank, **options))


def _add_backward_pass(schedule, block_id, mb_id, rank, options):
	for op_type in [OperationType.RECV_BACKWARD, OperationType.BACKWARD, OperationType.SEND_BACKWARD]:
		schedule.append(Operation(block_id, mb_id, op_type, rank, **options))


def generate_afab_schedule(placement, n_micro_batches, **options):
	"""
	All Forward All Backward as in GPipe https://arxiv.org/abs/1811.06965

	:param placement: device on which each block is placed
	:type placement: List[int]
	:param n_micro_batches: number of micro batches
	:type n_micro_batches: int
	:param **options: options to add to **each** operation of the schedule

	:return: a list containing the operations to execute for **all** processes
	:rtype: List[Operation]
	"""
	schedule = []
	n_stages = len(placement)
	n_devices = max(placement) + 1

	# All forward
	for rank in range(n_devices):
		ids = [i for i in range(len(placement)) if placement[i] == rank]
		for i in range(n_micro_batches):
			for id_ in ids:
				_add_forward_pass(schedule, id_, i, rank, options)

		# All backward
		for i in range(n_micro_batches):
			for id_ in reversed(ids):
				_add_backward_pass(schedule, id_, i, rank, options)

	assert len(schedule) == n_micro_batches * n_stages * 2 * 3
	return schedule


def generate_1f1b_schedule(placement, n_micro_batches, **options):
	"""
	One Forward One Backward as in PipeDream https://arxiv.org/abs/1806.03377

	:param placement: device on which each block is placed
	:type placement: List[int]
	:param n_micro_batches: number of micro batches
	:type n_micro_batches: int
	:param **options: options to add to **each** operation of the schedule

	:return: a list containing the operations to execute for **all** processes
	:rtype: List[Operation]
	"""
	schedule = []
	n_stages = len(placement)
	n_devices = int(max(placement)) + 1
	stages_per_device = n_stages // n_devices

	for rank in range(n_devices):
		fwds = [0] * stages_per_device
		bwds = [0] * stages_per_device

		i = 0
		b_f = 0
		# Warmup phase : each device can compute until the micro batch forward is finished (n_stages), but it can only start after it was forwarded through all the previous layers (rank)
		while i < (stages_per_device * n_micro_batches) and i < (n_stages - rank):
			i += 1
			_add_forward_pass(schedule, b_f * n_devices + rank, fwds[b_f], rank, options)
			fwds[b_f] += 1

			# each layer has time to compute n_devices micro batches before work arrives for the next layer
			# we always prioritize forward on the last possible layer
			# (also, we stop if all micro batches have been computed)
			if (i % n_devices) == 0 or (i % n_micro_batches) == 0:
				b_f = (b_f + 1) % stages_per_device

		# Number of forward passes computed before steady state
		state = i
		b_b = stages_per_device - 1  # last layer first

		# Steady state
		while i < (stages_per_device * n_micro_batches):
			i += 1
			_add_backward_pass(schedule, b_b * n_devices + rank, bwds[b_b], rank, options)
			bwds[b_b] += 1

			# Same as before, except that we can compute 2x less micro batches because half of the time is spent doing forwards
			if (i - state) % (n_devices // 2) == 0 or (i - state) % n_micro_batches == 0:
				b_b = (b_b - 1) % stages_per_device

			_add_forward_pass(schedule, b_f * n_devices + rank, fwds[b_f], rank, options)
			fwds[b_f] += 1

			if (i >= n_stages and i % (n_devices // 2) == 0) or (i % n_micro_batches) == 0:
				b_f = (b_f + 1) % stages_per_device

		while i < (
			stages_per_device * n_micro_batches * 2 - (stages_per_device * n_micro_batches - state)
		):
			i += 1
			_add_backward_pass(schedule, b_b * n_devices + rank, bwds[b_b], rank, options)
			bwds[b_b] += 1

			# Finish all backwards
			if (i - n_micro_batches - state) % (n_devices // 2) == 0 or (
				i - n_micro_batches - state
			) % n_micro_batches == 0:
				b_b = (b_b - 1) % stages_per_device

	return schedule


def generate_hanayo_schedule(placement, n_micro_batches, **options):
	"""
	Hanayo schedule as in https://dl.acm.org/doi/10.1145/3581784.3607073

	:param placement: device on which each block is placed
	:type placement: List[int]
	:param n_micro_batches: number of micro batches
	:type n_micro_batches: int
	:param **options: options to add to **each** operation of the schedule

	:return: a list containing the operations to execute for **all** processes
	:rtype: List[Operation]
	"""
	schedule = []
	n_devices = max(placement) + 1
	n_stages = len(placement)
	n_waves = n_stages // (n_devices * 2)

	ids = [[i for i in range(n_stages) if placement[i] == rank] for rank in range(n_devices)]

	done = [0 for _ in range(n_devices)]  # mb computed forward :)
	enod = [0 for _ in range(n_devices)]  # mb computed backward (:

	# Warmup
	for rank in range(n_devices):
		done[rank] = min(n_devices - rank, n_micro_batches)
		for i in range(done[rank]):
			_add_forward_pass(schedule, ids[rank][0], i, rank, options)

	for w in range(n_waves):
		for mb in range(n_micro_batches):
			for rank in reversed(range(n_devices)):
				_add_forward_pass(schedule, ids[rank][2 * w + 1], mb, rank, options)

				if done[rank] < n_micro_batches * n_waves:
					_add_forward_pass(
						schedule,
						ids[rank][2 * (done[rank] // n_micro_batches)],
						done[rank] % n_micro_batches,
						rank,
						options,
					)
					done[rank] += 1
				else:
					_add_backward_pass(
						schedule,
						ids[rank][-1 - 2 * (enod[rank] // n_micro_batches)],
						enod[rank] % n_micro_batches,
						rank,
						options,
					)
					enod[rank] += 1

	for w in range(n_waves):
		for mb in range(n_micro_batches):
			for rank in reversed(range(n_devices)):
				_add_backward_pass(schedule, ids[rank][-2 * (w + 1)], mb, rank, options)

				if enod[rank] < n_micro_batches * n_waves:
					_add_backward_pass(
						schedule,
						ids[rank][-1 - 2 * (enod[rank] // n_micro_batches)],
						enod[rank] % n_micro_batches,
						rank,
						options,
					)
					enod[rank] += 1

	return schedule
