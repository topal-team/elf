"""
Implementation of different state-of-the-art scheduling algorithms.
A schedule is a list of operations (see Operation) that will be executed in order by each device.
Every rank should generate the entire schedule for all ranks, in order to detect and fix cycles/deadlocks.
"""

import logging

from .scheduling import OpOptions, OperationType, Operation

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


def _add_backward_params(schedule, block_id, mb_id, rank, **options):
	schedule.append(Operation(block_id, mb_id, OperationType.BACKWARD_PARAMS, rank, **options))


def generate_afab_schedule(placement, n_micro_batches, signatures):
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
				_add_backward_params(schedule, id_, i, rank)

		for id_ in ids:
			schedule.append(Operation(id_, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule


def generate_1f1b_schedule(placement, n_micro_batches, signatures):
	""" """
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
			_add_backward_params(schedule, id_, bwds[b_b], rank)
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
			_add_backward_params(schedule, id_, bwds[b_b], rank)
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
	""" """
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
					_add_backward_params(schedule, id_, enod[rank] % n_micro_batches, rank)
					enod[rank] += 1

	for w in range(n_waves):
		for mb in range(n_micro_batches):
			for rank in reversed(range(n_devices)):
				id_ = ids[rank][-2 * (w + 1)]
				_add_backward_pass(schedule, placement, id_, mb, rank, signatures[id_])
				_add_backward_params(schedule, id_, mb, rank)

				if enod[rank] < n_micro_batches * n_waves:
					id_ = ids[rank][-1 - 2 * (enod[rank] // n_micro_batches)]
					_add_backward_pass(
						schedule, placement, id_, enod[rank] % n_micro_batches, rank, signatures[id_]
					)
					_add_backward_params(schedule, id_, enod[rank] % n_micro_batches, rank)
					enod[rank] += 1

	for i in range(n_stages):
		schedule.append(Operation(i, None, OperationType.ALL_REDUCE_PARAM_GRADS, placement[i]))

	return schedule


def generate_full_remat_schedule(placement, n_micro_batches, signatures):
	schedule = []
	n_devices = max(placement) + 1

	def remat_strategy(name, _):
		return name == ""

	# All ranks do afab
	for rank in range(n_devices - 1):
		ids = [i for i in range(len(placement)) if placement[i] == rank]
		# All forward
		for i in range(n_micro_batches):
			for id_ in ids:
				_add_forward_pass(
					schedule,
					placement,
					id_,
					i,
					rank,
					signatures[id_],
					**{OpOptions.REMAT_STRATEGY: remat_strategy},
				)
		# All backward
		for i in range(n_micro_batches):
			for id_ in reversed(ids):
				# schedule.append(
				# 	Operation(id_, i, OperationType.FORWARD, rank)
				# )  # recomputation happens here
				_add_backward_pass(schedule, placement, id_, i, rank, signatures[id_])
				_add_backward_params(schedule, id_, i, rank)

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
			_add_backward_params(schedule, id_, i, rank)

	for id_ in ids:
		schedule.append(Operation(id_, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule


def generate_zbh1_schedule(placement, n_micro_batches, signatures):
	schedule = []
	n_devices = len(set(placement))

	for block_id, rank in enumerate(placement):
		n_warmups = min(n_devices - block_id, n_micro_batches)
		f = 0
		b = 0
		w = 0

		for _ in range(n_warmups):
			_add_forward_pass(schedule, placement, block_id, f, rank, signatures[block_id])
			f += 1

		_add_backward_pass(schedule, placement, block_id, b, rank, signatures[block_id])
		b += 1

		for _ in range(rank):
			if f < n_micro_batches:
				_add_forward_pass(schedule, placement, block_id, f, rank, signatures[block_id])
				f += 1
			if b < n_micro_batches:
				_add_backward_pass(schedule, placement, block_id, b, rank, signatures[block_id])
				b += 1

		while f < n_micro_batches or b < n_micro_batches or w < n_micro_batches:
			if w < n_micro_batches:
				_add_backward_params(schedule, block_id, w, rank)
				w += 1
			if f < n_micro_batches:
				_add_forward_pass(schedule, placement, block_id, f, rank, signatures[block_id])
				f += 1
			if b < n_micro_batches:
				_add_backward_pass(schedule, placement, block_id, b, rank, signatures[block_id])
				b += 1

		schedule.append(Operation(block_id, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule


def generate_zbh2_schedule(placement, n_micro_batches, signatures):
	schedule = []
	n_devices = len(set(placement))

	for block_id, rank in enumerate(placement):
		# We don't support interleaving yet, only consider one block per rank
		# ids = [i for i in range(len(placement)) if placement[i] == rank]
		n_warmups = min(n_micro_batches, 2 * (n_devices - block_id) - 1)
		f = 0
		b = 0
		w = 0
		for i in range(n_warmups):
			_add_forward_pass(schedule, placement, block_id, f, rank, signatures[block_id])
			f += 1

		_add_backward_pass(schedule, placement, block_id, b, rank, signatures[block_id])
		b += 1

		for i in range(n_micro_batches - 1 - n_warmups):
			_add_forward_pass(schedule, placement, block_id, f, rank, signatures[block_id])
			f += 1

			_add_backward_pass(schedule, placement, block_id, b, rank, signatures[block_id])
			b += 1

		while f < n_micro_batches or b < n_micro_batches or w < n_micro_batches:
			if w < n_micro_batches:
				_add_backward_params(schedule, block_id, w, rank)
				w += 1
			if f < n_micro_batches:
				_add_forward_pass(schedule, placement, block_id, f, rank, signatures[block_id])
				f += 1
			if b < n_micro_batches:
				_add_backward_pass(schedule, placement, block_id, b, rank, signatures[block_id])
				b += 1

		schedule.append(Operation(block_id, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule


def generate_zbv_schedule(placement, n_micro_batches, signatures):
	""" """
	schedule = []
	n_devices = max(placement) + 1
	stages_per_device = len(placement) // n_devices

	if n_micro_batches != n_devices * 2:
		logger.warning(
			f"ZBV schedule is only tested for nmb = 2 * n_devices, got {n_micro_batches}. The schedule may be incorrect."
		)

	for rank in range(n_devices):
		fs = [0] * stages_per_device
		bs = [0] * stages_per_device
		ws = [0] * stages_per_device

		ids = [i for i in range(len(placement)) if placement[i] == rank]
		n_warmups = min(n_micro_batches, 2 * (n_devices - rank) - 1)

		# Warmup, phase 1: first stage
		current = 0
		for _ in range(n_warmups):
			_add_forward_pass(
				schedule, placement, ids[current], fs[current], rank, signatures[ids[current]]
			)
			fs[current] += 1

		# Warmup, phase 2: alternating stages
		current = 1
		for _ in range(2 * n_devices - n_warmups - 1):
			id_ = ids[current]
			_add_forward_pass(schedule, placement, id_, fs[current], rank, signatures[id_])
			fs[current] += 1
			current = 1 - current

		# Steady, phase 1: last stage
		current = 1
		for _ in range(n_devices - rank):
			_add_forward_pass(
				schedule, placement, ids[current], fs[current], rank, signatures[ids[current]]
			)
			fs[current] += 1
			_add_backward_pass(
				schedule, placement, ids[current], bs[current], rank, signatures[ids[current]]
			)
			_add_backward_params(schedule, ids[current], bs[current], rank)
			bs[current] += 1
			schedule[-1].mb_id = ws[current]
			ws[current] += 1

		# Steady, phase 2: alternating stages
		for _ in range(rank + 1):
			current = 0
			_add_forward_pass(
				schedule, placement, ids[current], fs[current], rank, signatures[ids[current]]
			)
			fs[current] += 1
			_add_backward_pass(
				schedule, placement, ids[current], bs[current], rank, signatures[ids[current]]
			)
			_add_backward_params(schedule, ids[current], bs[current], rank)
			bs[current] += 1
			schedule[-1].mb_id = ws[current]
			ws[current] += 1

			current = 1
			_add_forward_pass(
				schedule, placement, ids[current], fs[current], rank, signatures[ids[current]]
			)
			fs[current] += 1
			_add_backward_pass(
				schedule, placement, ids[current], bs[current], rank, signatures[ids[current]]
			)
			_add_backward_params(schedule, ids[current], bs[current], rank)
			bs[current] += 1
			schedule[-1].mb_id = ws[current]
			ws[current] += 1

		# Irregularity in the FBW pattern
		for _ in range(n_devices - rank - 1):
			current = 0
			_add_backward_pass(
				schedule, placement, ids[current], bs[current], rank, signatures[ids[current]]
			)
			_add_backward_params(schedule, ids[current], bs[current], rank)
			bs[current] += 1
			schedule[-1].mb_id = ws[current]
			ws[current] += 1

			current = 1
			_add_forward_pass(
				schedule, placement, ids[current], fs[current], rank, signatures[ids[current]]
			)
			fs[current] += 1
			_add_backward_pass(
				schedule, placement, ids[current], bs[current], rank, signatures[ids[current]]
			)
			_add_backward_params(schedule, ids[current], bs[current], rank)
			bs[current] += 1
			schedule[-1].mb_id = ws[current]
			ws[current] += 1

		# Cooldown: inverse warmup phase
		n_cooldowns = 2 * n_devices - n_warmups
		current = 0
		for _ in range(n_cooldowns):
			_add_backward_pass(
				schedule, placement, ids[current], bs[current], rank, signatures[ids[current]]
			)
			bs[current] += 1
			current = 1 - current

		for _ in range(n_devices - rank - 1):
			current = 0
			_add_backward_params(schedule, ids[current], ws[current], rank)
			ws[current] += 1
			_add_backward_pass(
				schedule, placement, ids[current], bs[current], rank, signatures[ids[current]]
			)
			bs[current] += 1

		for _ in range(rank + 1):
			_add_backward_params(schedule, ids[0], ws[0], rank)
			ws[0] += 1
		for _ in range(rank):
			_add_backward_params(schedule, ids[1], ws[1], rank)
			ws[1] += 1

		for id_ in ids:
			schedule.append(Operation(id_, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

		for i in range(stages_per_device):
			assert fs[i] == bs[i] == ws[i] == n_micro_batches, (
				f"Rank {rank}, stage {i}: f = {fs[i]}, b = {bs[i]}, w = {ws[i]} (expected {n_micro_batches})"
			)

	return schedule


def generate_inference_schedule(placement, n_micro_batches, signatures):
	schedule = []

	for id_ in range(len(placement)):
		for mb in range(n_micro_batches):
			_add_forward_pass(schedule, placement, id_, mb, placement[id_], signatures[id_])
			if id_ == len(placement) - 1:
				schedule.pop(-1)  # remove loss forward

	return schedule


class FixedSchedule:
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
		self.schedule_dict = schedule_dict

	def __call__(self, placement, n_micro_batches, signatures):
		schedule = []
		order = self.schedule_dict["order"]

		for optype, block_id, mb_id in order:
			block_id = int(block_id)
			mb_id = int(mb_id)
			match optype:
				case "f":
					_add_forward_pass(
						schedule, placement, block_id, mb_id, placement[block_id], signatures[block_id]
					)
				case "b":
					_add_backward_pass(
						schedule, placement, block_id, mb_id, placement[block_id], signatures[block_id]
					)
				case "w":
					_add_backward_params(schedule, block_id, mb_id, placement[block_id])

		for block_id in range(len(placement)):
			schedule.append(
				Operation(block_id, None, OperationType.ALL_REDUCE_PARAM_GRADS, placement[block_id])
			)

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
					b += 1
				case "w":
					_add_backward_params(schedule, rank, w, rank)
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
					to_recompute_bact.append(b)
					b += 1
				case "g":
					_add_backward_pass(
						schedule, placement, rank, b, rank, signatures[rank], **{OpOptions.DEL_GRAD_BW: True}
					)
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
