import pytest

from elf.registry import SCHEDULERS
from elf.scheduling import Operation, OperationType, add_comms
from elf.partitioners.utils import sequential_signatures


@pytest.mark.unit
def test_afab():
	placement = [0, 1]
	n_micro_batches = 2

	schedule = SCHEDULERS["afab"](placement, n_micro_batches)
	expected_schedule = [
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 1, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 0, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, 1, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 1, OperationType.BACKWARD_PARAMS, 0),
		Operation(1, 0, OperationType.FORWARD, 1),
		Operation(1, 0, OperationType.LOSS_FORWARD, 1),
		Operation(1, 1, OperationType.FORWARD, 1),
		Operation(1, 1, OperationType.LOSS_FORWARD, 1),
		Operation(1, 0, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 0, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 0, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, 1, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 1, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 1, OperationType.BACKWARD_PARAMS, 1),
	]

	for op, expected_op in zip(schedule, expected_schedule):
		assert op == expected_op


@pytest.mark.unit
def test_1f1b():
	placement = [0, 1]
	n_micro_batches = 4

	schedule = SCHEDULERS["1f1b"](placement, n_micro_batches)
	schedule0 = list(filter(lambda op: op.rank == 0, schedule))

	assert schedule0 == [
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 1, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 0, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, 2, OperationType.FORWARD, 0),
		Operation(0, 1, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 1, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, 3, OperationType.FORWARD, 0),
		Operation(0, 2, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 2, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, 3, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 3, OperationType.BACKWARD_PARAMS, 0),
	]

	schedule1 = list(filter(lambda op: op.rank == 1, schedule))

	assert schedule1 == [
		Operation(1, 0, OperationType.FORWARD, 1),
		Operation(1, 0, OperationType.LOSS_FORWARD, 1),
		Operation(1, 0, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 0, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 0, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, 1, OperationType.FORWARD, 1),
		Operation(1, 1, OperationType.LOSS_FORWARD, 1),
		Operation(1, 1, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 1, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 1, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, 2, OperationType.FORWARD, 1),
		Operation(1, 2, OperationType.LOSS_FORWARD, 1),
		Operation(1, 2, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 2, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 2, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, 3, OperationType.FORWARD, 1),
		Operation(1, 3, OperationType.LOSS_FORWARD, 1),
		Operation(1, 3, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 3, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 3, OperationType.BACKWARD_PARAMS, 1),
	]


def check_validity(schedule, placement, n_micro_batches):
	n_stages = len(placement)

	# Count the number of each type of operation
	fwds = [op for op in schedule if op.op == OperationType.FORWARD]
	bwds_inputs = [op for op in schedule if op.op == OperationType.BACKWARD_INPUTS]
	bwds_params = [op for op in schedule if op.op == OperationType.BACKWARD_PARAMS]

	send_fwds = [op for op in schedule if op.op == OperationType.SEND_FORWARD]
	send_bwds = [op for op in schedule if op.op == OperationType.SEND_BACKWARD]
	recv_fwds = [op for op in schedule if op.op == OperationType.RECV_FORWARD]
	recv_bwds = [op for op in schedule if op.op == OperationType.RECV_BACKWARD]
	assert (
		len(fwds)
		== len(bwds_inputs)
		== len(bwds_params)
		== len(send_fwds)
		== len(send_bwds)
		== len(recv_fwds)
		== len(recv_bwds)
		== (n_stages * n_micro_batches)
	)

	loss_fwds = [op for op in schedule if op.op == OperationType.LOSS_FORWARD]
	loss_bwds = [op for op in schedule if op.op == OperationType.LOSS_BACKWARD]
	assert len(loss_fwds) == len(loss_bwds) == n_micro_batches

	# All stages have the same number of forward and backward ops
	for i in range(n_stages):
		fwds_mb = [op for op in fwds if op.block_id == i]
		bwds_inputs_mb = [op for op in bwds_inputs if op.block_id == i]
		bwds_params_mb = [op for op in bwds_params if op.block_id == i]

		assert len(fwds_mb) == len(bwds_inputs_mb) == len(bwds_params_mb) == n_micro_batches

	# All micro batches have the same number of forward and backward ops
	for i in range(n_micro_batches):
		fwds_stage = [op for op in fwds if op.mb_id == i]
		bwds_inputs_stage = [op for op in bwds_inputs if op.mb_id == i]
		bwds_params_stage = [op for op in bwds_params if op.mb_id == i]

		assert len(fwds_stage) == len(bwds_inputs_stage) == len(bwds_params_stage) == n_stages

	def check_order(block_id, mb_id, is_last=False):
		ops = [OperationType.FORWARD]
		if is_last:
			ops.extend([OperationType.LOSS_FORWARD, OperationType.LOSS_BACKWARD])
		ops.extend([OperationType.BACKWARD_INPUTS, OperationType.BACKWARD_PARAMS])
		for op in schedule:
			if op.block_id != block_id or op.mb_id != mb_id or op.op not in ops:
				continue
			assert op.op == ops.pop(0)

	# Check that the order is always forward/backward_inputs/bparams
	for mb_id in range(n_micro_batches):
		for block_id in range(n_stages):
			check_order(block_id, mb_id, is_last=block_id == n_stages - 1)

	# All reduce param grads should be after all backward ops
	for block_id in range(n_stages):
		ops = [op for op in schedule if op.block_id == block_id]
		n_bwds = 0
		for op in ops:
			if op.op == OperationType.BACKWARD_PARAMS:
				n_bwds += 1
			elif op.op == OperationType.ALL_REDUCE_PARAM_GRADS:
				assert n_bwds == n_micro_batches

	# Number of all reduce param grads should be equal to the number of stages
	n_all_reduce_param_grads = len(
		[op for op in schedule if op.op == OperationType.ALL_REDUCE_PARAM_GRADS]
	)
	assert n_all_reduce_param_grads == n_stages


@pytest.mark.unit
def test_schedulers():
	setups = {
		"gpipe": {
			"placements": [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]],
			"n_micro_batches": [2, 4, 8],
		},
		"1f1b": {
			"placements": [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]],
			"n_micro_batches": [2, 4, 8],
		},
		"hanayo": {
			"placements": [
				[0, 1, 2, 3, 3, 2, 1, 0],
				[0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
				[0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0],
			],
			"n_micro_batches": [2, 4, 8],
		},
		"zbh1": {"placements": [[0, 1], [0, 1, 2, 3]], "n_micro_batches": [2, 4, 8]},
		"zbh2": {"placements": [[0, 1], [0, 1, 2, 3]], "n_micro_batches": [2, 4, 8]},
		"zbv": {
			# Only tested with 8 micro batches right now
			"placements": [[0, 1, 2, 3, 3, 2, 1, 0]],
			"n_micro_batches": [8],
		},
	}

	for scheduler_name, setup in setups.items():
		scheduler = SCHEDULERS[scheduler_name]
		for placement in setup["placements"]:
			for n_micro_batches in setup["n_micro_batches"]:
				signatures = sequential_signatures(placement)
				schedule = scheduler(placement, n_micro_batches)
				schedule = add_comms(schedule, signatures)
				try:
					check_validity(schedule, placement, n_micro_batches)
				except Exception as e:
					pytest.fail(
						f"Scheduler '{scheduler_name}' failed with placement={placement}, n_micro_batches={n_micro_batches}: {e}"
					)


@pytest.mark.unit
def test_inference_scheduler():
	"""Test that inference scheduler produces no backward operations."""
	placement = [0, 1, 2, 3]
	n_micro_batches = 4

	scheduler = SCHEDULERS["inference"]
	schedule = scheduler(placement, n_micro_batches)

	backward_ops = [
		op
		for op in schedule
		if op.op
		in [OperationType.BACKWARD_INPUTS, OperationType.BACKWARD_PARAMS, OperationType.LOSS_BACKWARD]
	]

	assert len(backward_ops) == 0, "Inference schedule should not have backward operations"


@pytest.mark.unit
def test_add_comms():
	schedule = [
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 1, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 0, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, 1, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 1, OperationType.BACKWARD_PARAMS, 0),
	]
	signatures = sequential_signatures([0, 1])
	schedule = add_comms(schedule, signatures)
	assert len([op for op in schedule if op.op == OperationType.ALL_REDUCE_PARAM_GRADS]) == 1

	# Check that signatures sources and targets are respected
	send_fwd_ops = [op for op in schedule if op.op == OperationType.SEND_FORWARD]
	recv_fwd_ops = [op for op in schedule if op.op == OperationType.RECV_FORWARD]
	send_bwd_ops = [op for op in schedule if op.op == OperationType.SEND_BACKWARD]
	recv_bwd_ops = [op for op in schedule if op.op == OperationType.RECV_BACKWARD]

	# For sequential signatures [0, 1], block 0 should send forward to block 1
	assert len(send_fwd_ops) == 2  # 2 micro-batches
	for op in send_fwd_ops:
		assert op.block_id == 0
		assert op.rank == 0

	# Block 1 should receive forward from block 0
	assert len(recv_fwd_ops) == 2  # 2 micro-batches
	for op in recv_fwd_ops:
		assert op.block_id == 0
		assert op.rank == 0

	# Block 1 should send backward to block 0
	assert len(send_bwd_ops) == 2  # 2 micro-batches
	for op in send_bwd_ops:
		assert op.block_id == 0
		assert op.rank == 0

	# Block 0 should receive backward from block 1
	assert len(recv_bwd_ops) == 2  # 2 micro-batches
	for op in recv_bwd_ops:
		assert op.block_id == 0
		assert op.rank == 0
