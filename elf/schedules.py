"""
Manage different types of schedule.
A schedule is a list of operations (see Operation) that will be executed in order by each device.
Every rank should generate the entire schedule for all ranks, in order to detect and fix cycles/deadlocks.
"""

from .scheduling import OpOptions, OperationType, Operation
import logging

logger = logging.getLogger("schedules")


def _add_forward_pass(schedule, placement, block_id, mb_id, rank, signature, **options):
	sources = sorted(signature.get_all_sources(), reverse=True)
	destinations = sorted(signature.get_all_targets())
	for src in sources:
		schedule.append(
			Operation(block_id, mb_id, OperationType.RECV_FORWARD, rank, src=src, **options)
		)
	schedule.append(Operation(block_id, mb_id, OperationType.FORWARD, rank, **options))
	for dst in destinations:
		schedule.append(
			Operation(block_id, mb_id, OperationType.SEND_FORWARD, rank, dst=dst, **options)
		)

	if block_id == len(placement) - 1:
		schedule.append(Operation(block_id, mb_id, OperationType.LOSS_FORWARD, rank, **options))


def _add_backward_pass(schedule, placement, block_id, mb_id, rank, signature, **options):
	# Reverse order, see above
	sources = sorted(signature.get_all_sources(), reverse=True)
	destinations = sorted(signature.get_all_targets())
	if block_id == len(placement) - 1:
		schedule.append(Operation(block_id, mb_id, OperationType.LOSS_BACKWARD, rank, **options))

	# backward swaps the sources and destinations
	for src in destinations:
		schedule.append(
			Operation(block_id, mb_id, OperationType.RECV_BACKWARD, rank, src=src, **options)
		)
	schedule.append(Operation(block_id, mb_id, OperationType.BACKWARD_INPUTS, rank, **options))
	for dst in sources:
		schedule.append(
			Operation(block_id, mb_id, OperationType.SEND_BACKWARD, rank, dst=dst, **options)
		)
	schedule.append(Operation(block_id, mb_id, OperationType.BACKWARD_PARAMS, rank, **options))


def generate_afab_schedule(placement, n_micro_batches, signatures):
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
	n_devices = max(placement) + 1

	for rank in range(n_devices):
		ids = [i for i in range(len(placement)) if placement[i] == rank]
		# All forward
		for i in range(n_micro_batches):
			for id_ in ids:
				_add_forward_pass(schedule, placement, id_, i, rank, signatures[id_])
		# All backward
		for i in range(n_micro_batches):
			for id_ in reversed(ids):
				_add_backward_pass(schedule, placement, id_, i, rank, signatures[id_])

		for id_ in ids:
			schedule.append(Operation(id_, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule


def generate_1f1b_schedule(placement, n_micro_batches, signatures):
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
			id_ = b_f * n_devices + rank
			_add_forward_pass(schedule, placement, id_, fwds[b_f], rank, signatures[id_])
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
			id_ = b_b * n_devices + rank
			_add_backward_pass(schedule, placement, id_, bwds[b_b], rank, signatures[id_])
			bwds[b_b] += 1

			# Same as before, except that we can compute 2x less micro batches because half of the time is spent doing forwards
			if (i - state) % (n_devices // 2) == 0 or (i - state) % n_micro_batches == 0:
				b_b = (b_b - 1) % stages_per_device

			id_ = b_f * n_devices + rank
			_add_forward_pass(schedule, placement, id_, fwds[b_f], rank, signatures[id_])
			fwds[b_f] += 1

			if (i >= n_stages and i % (n_devices // 2) == 0) or (i % n_micro_batches) == 0:
				b_f = (b_f + 1) % stages_per_device

		while i < (
			stages_per_device * n_micro_batches * 2 - (stages_per_device * n_micro_batches - state)
		):
			i += 1
			id_ = b_b * n_devices + rank
			_add_backward_pass(schedule, placement, id_, bwds[b_b], rank, signatures[id_])
			bwds[b_b] += 1

			# Finish all backwards
			if (i - n_micro_batches - state) % (n_devices // 2) == 0 or (
				i - n_micro_batches - state
			) % n_micro_batches == 0:
				b_b = (b_b - 1) % stages_per_device

	for i in range(n_stages):
		schedule.append(Operation(i, None, OperationType.ALL_REDUCE_PARAM_GRADS, placement[i]))

	return schedule


def generate_hanayo_schedule(placement, n_micro_batches, signatures):
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
			id_ = ids[rank][0]
			_add_forward_pass(schedule, placement, id_, i, rank, signatures[id_])

	for w in range(n_waves):
		for mb in range(n_micro_batches):
			for rank in reversed(range(n_devices)):
				id_ = ids[rank][2 * w + 1]
				_add_forward_pass(schedule, placement, id_, mb, rank, signatures[id_])

				if done[rank] < n_micro_batches * n_waves:
					id_ = ids[rank][2 * (done[rank] // n_micro_batches)]
					_add_forward_pass(
						schedule, placement, id_, done[rank] % n_micro_batches, rank, signatures[id_]
					)
					done[rank] += 1
				else:
					id_ = ids[rank][-1 - 2 * (enod[rank] // n_micro_batches)]
					_add_backward_pass(
						schedule, placement, id_, enod[rank] % n_micro_batches, rank, signatures[id_]
					)
					enod[rank] += 1

	for w in range(n_waves):
		for mb in range(n_micro_batches):
			for rank in reversed(range(n_devices)):
				id_ = ids[rank][-2 * (w + 1)]
				_add_backward_pass(schedule, placement, id_, mb, rank, signatures[id_])

				if enod[rank] < n_micro_batches * n_waves:
					id_ = ids[rank][-1 - 2 * (enod[rank] // n_micro_batches)]
					_add_backward_pass(
						schedule, placement, id_, enod[rank] % n_micro_batches, rank, signatures[id_]
					)
					enod[rank] += 1

	for i in range(n_stages):
		schedule.append(Operation(i, None, OperationType.ALL_REDUCE_PARAM_GRADS, placement[i]))

	return schedule


def generate_full_remat_schedule(placement, n_micro_batches, signatures):
	schedule = []
	n_devices = max(placement) + 1

	# All ranks do afab
	for rank in range(n_devices - 1):
		ids = [i for i in range(len(placement)) if placement[i] == rank]
		# All forward
		for i in range(n_micro_batches):
			for id_ in ids:
				_add_forward_pass(
					schedule, placement, id_, i, rank, signatures[id_], **{OpOptions.SAVE: False}
				)
		# All backward
		for i in range(n_micro_batches):
			for id_ in reversed(ids):
				# schedule.append(
				# 	Operation(id_, i, OperationType.FORWARD, rank)
				# )  # recomputation happens here
				_add_backward_pass(schedule, placement, id_, i, rank, signatures[id_])

		for id_ in ids:
			schedule.append(Operation(id_, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	# Last rank does 1f1b
	rank = n_devices - 1
	ids = [i for i in range(len(placement)) if placement[i] == rank]
	for i in range(n_micro_batches):
		for id_ in ids:
			_add_forward_pass(schedule, placement, id_, i, rank, signatures[id_])

		for id_ in reversed(ids):
			_add_backward_pass(schedule, placement, id_, i, rank, signatures[id_])

	for id_ in ids:
		schedule.append(Operation(id_, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule


def generate_zbh1_schedule(placement, n_micro_batches, signatures):
	schedule = []
	n_devices = max(placement) + 1

	for rank in range(n_devices):
		n_warmups = min(n_devices - rank, n_micro_batches)
		f = 0
		b = 0
		w = 0

		for _ in range(n_warmups):
			_add_forward_pass(schedule, placement, rank, f, rank, signatures[rank])
			f += 1

		_add_backward_pass(schedule, placement, rank, b, rank, signatures[rank])
		schedule.pop(-1)  # hack: remove the backward wrt weights
		b += 1

		for _ in range(rank):
			_add_forward_pass(schedule, placement, rank, f, rank, signatures[rank])
			f += 1
			_add_backward_pass(schedule, placement, rank, b, rank, signatures[rank])
			schedule.pop(-1)  # hack: remove the backward wrt weights
			b += 1

		while f < n_micro_batches or b < n_micro_batches or w < n_micro_batches:
			if w < n_micro_batches:
				schedule.append(Operation(rank, w, OperationType.BACKWARD_PARAMS, rank))
				w += 1
			if f < n_micro_batches:
				_add_forward_pass(schedule, placement, rank, f, rank, signatures[rank])
				f += 1
			if b < n_micro_batches:
				_add_backward_pass(schedule, placement, rank, b, rank, signatures[rank])
				schedule.pop(-1)  # hack: remove the backward wrt weights
				b += 1

		schedule.append(Operation(rank, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule


def generate_zbh2_schedule(placement, n_micro_batches, signatures):
	schedule = []
	n_devices = max(placement) + 1

	for rank in range(n_devices):
		# We don't support interleaving yet, only consider one block per rank
		# ids = [i for i in range(len(placement)) if placement[i] == rank]
		n_warmups = min(n_micro_batches, 2 * (n_devices - rank) - 1)
		f = 0
		b = 0
		w = 0
		for i in range(n_warmups):
			_add_forward_pass(schedule, placement, rank, f, rank, signatures[rank])
			f += 1

		_add_backward_pass(schedule, placement, rank, b, rank, signatures[rank])
		schedule.pop(-1)  # hack: remove the backward wrt weights
		b += 1

		for i in range(n_micro_batches - 1 - n_warmups):
			_add_forward_pass(schedule, placement, rank, f, rank, signatures[rank])
			f += 1

			_add_backward_pass(schedule, placement, rank, b, rank, signatures[rank])
			schedule.pop(-1)  # hack: remove the backward wrt weights
			b += 1

		while f < n_micro_batches or b < n_micro_batches or w < n_micro_batches:
			if w < n_micro_batches:
				schedule.append(Operation(rank, w, OperationType.BACKWARD_PARAMS, rank))
				w += 1
			if f < n_micro_batches:
				_add_forward_pass(schedule, placement, rank, f, rank, signatures[rank])
				f += 1
			if b < n_micro_batches:
				_add_backward_pass(schedule, placement, rank, b, rank, signatures[rank])
				schedule.pop(-1)  # hack: remove the backward wrt weights
				b += 1

		schedule.append(Operation(rank, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule


def generate_inference_schedule(placement, n_micro_batches, signatures):
	schedule = []

	for id_ in range(len(placement)):
		for mb in range(n_micro_batches):
			_add_forward_pass(schedule, placement, id_, mb, placement[id_], signatures[id_])
			if id_ == len(placement) - 1:
				schedule.pop(-1)  # remove loss forward

	return schedule


def schedule_from_str(schedule_str, placement, signatures):
	"""
	Generate a schedule from a string representation. All communications are automatically handled. Micro batches are computed in ascending order. An AllReduce operation is added at the end of the schedule.
	Expected representation:
		schedule_str = [
			"fbwfbw",
			"ffbbww"
		]
	This example will be executed on 2 ranks (2 strings).
	- f: forward and save all
	- F: forward and save none
	- r: recompute forward (and save all)
	- b: backward for inputs and save all
	- g: backward for inputs and save gradients
	- a: backward for inputs and save activations
	- B: backward for inputs and save none
	- w: backward for weights
	"""
	schedule = []
	for rank, rank_sched in enumerate(schedule_str):
		f = 0
		b = 0
		w = 0
		to_recompute_forward = []
		to_recompute_bact = []
		to_recompute_bgrads = []
		for op in rank_sched:
			match op:
				case "f":
					_add_forward_pass(schedule, placement, rank, f, rank, signatures[rank])
					f += 1
				case "b":
					_add_backward_pass(schedule, placement, rank, b, rank, signatures[rank])
					schedule.pop(-1)  # hack: remove the backward wrt weights
					b += 1
				case "w":
					schedule.append(Operation(rank, w, OperationType.BACKWARD_PARAMS, rank))
					w += 1
				case "F":
					to_recompute_forward.append(f)
					_add_forward_pass(
						schedule, placement, rank, f, rank, signatures[rank], **{OpOptions.SAVE: False}
					)
					f += 1
				case "r":
					mb_id = to_recompute_forward.pop(0)
					schedule.append(Operation(rank, mb_id, OperationType.FORWARD, rank))
				case "a":
					_add_backward_pass(
						schedule, placement, rank, b, rank, signatures[rank], **{OpOptions.DEL_ACT_BW: True}
					)
					schedule.pop(-1)
					to_recompute_bact.append(b)
					b += 1
				case "g":
					_add_backward_pass(
						schedule, placement, rank, b, rank, signatures[rank], **{OpOptions.DEL_GRAD_BW: True}
					)
					schedule.pop(-1)
					to_recompute_bgrads.append(b)
					b += 1
				case "B":
					_add_backward_pass(
						schedule,
						placement,
						rank,
						b,
						rank,
						signatures[rank],
						**{OpOptions.DEL_ACT_BW: True, OpOptions.DEL_GRAD_BW: True},
					)
					schedule.pop(-1)  # hack: remove the backward wrt weights
					to_recompute_bact.append(b)
					to_recompute_bgrads.append(b)
					b += 1
				case "p":
					mb_id = to_recompute_bact.pop(0)
					schedule.append(
						Operation(rank, mb_id, OperationType.FORWARD, rank), **{OpOptions.REMAT_ACT_BW: True}
					)
				case "P":
					mb_id = to_recompute_bgrads.pop(0)
					schedule.append(
						Operation(rank, mb_id, OperationType.BACKWARD_INPUTS, rank),
						**{OpOptions.REMAT_GRAD_BW: True},
					)
				case "AR":
					pass  # we will add anyway later

		schedule.append(Operation(rank, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule
