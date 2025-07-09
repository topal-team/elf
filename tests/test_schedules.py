from elf.partitioners.utils import Signature
from elf.scheduling import Operation, OperationType
from elf.registry import SCHEDULERS
import pytest


@pytest.mark.single
def test_afab():
	placement = [0, 1]
	n_micro_batches = 2
	sources = [[None], [0]]
	destinations = [[[1]], [[None]]]

	schedule = SCHEDULERS["afab"](
		placement,
		n_micro_batches,
		signatures=[Signature([""], [""], src, dst) for src, dst in zip(sources, destinations)],
	)

	assert schedule == [
		Operation(0, 0, OperationType.RECV_FORWARD, 0, src=None),
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.SEND_FORWARD, 0, dst=1),
		Operation(0, 1, OperationType.RECV_FORWARD, 0, src=None),
		Operation(0, 1, OperationType.FORWARD, 0),
		Operation(0, 1, OperationType.SEND_FORWARD, 0, dst=1),
		Operation(0, 0, OperationType.RECV_BACKWARD, 0, src=1),
		Operation(0, 0, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 0, OperationType.SEND_BACKWARD, 0, dst=None),
		Operation(0, 0, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, 1, OperationType.RECV_BACKWARD, 0, src=1),
		Operation(0, 1, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 1, OperationType.SEND_BACKWARD, 0, dst=None),
		Operation(0, 1, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, None, OperationType.ALL_REDUCE_PARAM_GRADS, 0),
		Operation(1, 0, OperationType.RECV_FORWARD, 1, src=0),
		Operation(1, 0, OperationType.FORWARD, 1),
		Operation(1, 0, OperationType.SEND_FORWARD, 1, dst=None),
		Operation(1, 0, OperationType.LOSS_FORWARD, 1),
		Operation(1, 1, OperationType.RECV_FORWARD, 1, src=0),
		Operation(1, 1, OperationType.FORWARD, 1),
		Operation(1, 1, OperationType.SEND_FORWARD, 1, dst=None),
		Operation(1, 1, OperationType.LOSS_FORWARD, 1),
		Operation(1, 0, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 0, OperationType.RECV_BACKWARD, 1, src=None),
		Operation(1, 0, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 0, OperationType.SEND_BACKWARD, 1, dst=0),
		Operation(1, 0, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, 1, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 1, OperationType.RECV_BACKWARD, 1, src=None),
		Operation(1, 1, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 1, OperationType.SEND_BACKWARD, 1, dst=0),
		Operation(1, 1, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, None, OperationType.ALL_REDUCE_PARAM_GRADS, 1),
	]


@pytest.mark.single
def test_1f1b():
	placement = [0, 1]
	n_micro_batches = 4
	sources = [[None], [0]]
	destinations = [[[1]], [[None]]]

	schedule = SCHEDULERS["1f1b"](
		placement,
		n_micro_batches,
		signatures=[Signature([""], [""], src, dst) for src, dst in zip(sources, destinations)],
	)
	schedule0 = list(filter(lambda op: op.rank == 0, schedule))

	assert schedule0 == [
		Operation(0, 0, OperationType.RECV_FORWARD, 0, src=None),
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.SEND_FORWARD, 0, dst=1),
		Operation(0, 1, OperationType.RECV_FORWARD, 0, src=None),
		Operation(0, 1, OperationType.FORWARD, 0),
		Operation(0, 1, OperationType.SEND_FORWARD, 0, dst=1),
		Operation(0, 0, OperationType.RECV_BACKWARD, 0, src=1),
		Operation(0, 0, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 0, OperationType.SEND_BACKWARD, 0, dst=None),
		Operation(0, 0, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, 2, OperationType.RECV_FORWARD, 0, src=None),
		Operation(0, 2, OperationType.FORWARD, 0),
		Operation(0, 2, OperationType.SEND_FORWARD, 0, dst=1),
		Operation(0, 1, OperationType.RECV_BACKWARD, 0, src=1),
		Operation(0, 1, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 1, OperationType.SEND_BACKWARD, 0, dst=None),
		Operation(0, 1, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, 3, OperationType.RECV_FORWARD, 0, src=None),
		Operation(0, 3, OperationType.FORWARD, 0),
		Operation(0, 3, OperationType.SEND_FORWARD, 0, dst=1),
		Operation(0, 2, OperationType.RECV_BACKWARD, 0, src=1),
		Operation(0, 2, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 2, OperationType.SEND_BACKWARD, 0, dst=None),
		Operation(0, 2, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, 3, OperationType.RECV_BACKWARD, 0, src=1),
		Operation(0, 3, OperationType.BACKWARD_INPUTS, 0),
		Operation(0, 3, OperationType.SEND_BACKWARD, 0, dst=None),
		Operation(0, 3, OperationType.BACKWARD_PARAMS, 0),
		Operation(0, None, OperationType.ALL_REDUCE_PARAM_GRADS, 0),
	]

	schedule1 = list(filter(lambda op: op.rank == 1, schedule))

	# Send/Recv are swapped on odd devices to avoid deadlocks :)
	assert schedule1 == [
		Operation(1, 0, OperationType.RECV_FORWARD, 1, src=0),
		Operation(1, 0, OperationType.FORWARD, 1),
		Operation(1, 0, OperationType.SEND_FORWARD, 1, dst=None),
		Operation(1, 0, OperationType.LOSS_FORWARD, 1),
		Operation(1, 0, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 0, OperationType.RECV_BACKWARD, 1, src=None),
		Operation(1, 0, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 0, OperationType.SEND_BACKWARD, 1, dst=0),
		Operation(1, 0, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, 1, OperationType.RECV_FORWARD, 1, src=0),
		Operation(1, 1, OperationType.FORWARD, 1),
		Operation(1, 1, OperationType.SEND_FORWARD, 1, dst=None),
		Operation(1, 1, OperationType.LOSS_FORWARD, 1),
		Operation(1, 1, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 1, OperationType.RECV_BACKWARD, 1, src=None),
		Operation(1, 1, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 1, OperationType.SEND_BACKWARD, 1, dst=0),
		Operation(1, 1, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, 2, OperationType.RECV_FORWARD, 1, src=0),
		Operation(1, 2, OperationType.FORWARD, 1),
		Operation(1, 2, OperationType.SEND_FORWARD, 1, dst=None),
		Operation(1, 2, OperationType.LOSS_FORWARD, 1),
		Operation(1, 2, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 2, OperationType.RECV_BACKWARD, 1, src=None),
		Operation(1, 2, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 2, OperationType.SEND_BACKWARD, 1, dst=0),
		Operation(1, 2, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, 3, OperationType.RECV_FORWARD, 1, src=0),
		Operation(1, 3, OperationType.FORWARD, 1),
		Operation(1, 3, OperationType.SEND_FORWARD, 1, dst=None),
		Operation(1, 3, OperationType.LOSS_FORWARD, 1),
		Operation(1, 3, OperationType.LOSS_BACKWARD, 1),
		Operation(1, 3, OperationType.RECV_BACKWARD, 1, src=None),
		Operation(1, 3, OperationType.BACKWARD_INPUTS, 1),
		Operation(1, 3, OperationType.SEND_BACKWARD, 1, dst=0),
		Operation(1, 3, OperationType.BACKWARD_PARAMS, 1),
		Operation(1, None, OperationType.ALL_REDUCE_PARAM_GRADS, 1),
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
		ops = [OperationType.RECV_FORWARD, OperationType.FORWARD, OperationType.SEND_FORWARD]
		if is_last:
			ops.extend([OperationType.LOSS_FORWARD, OperationType.LOSS_BACKWARD])
		ops.extend(
			[OperationType.RECV_BACKWARD, OperationType.BACKWARD_INPUTS, OperationType.SEND_BACKWARD]
		)
		for op in schedule:
			if op.block_id != block_id or op.mb_id != mb_id or op.op not in ops:
				continue
			assert op.op == ops.pop(0)

	# Check that the order is always recv/forward/send/recv/backward_inputs/send (bparams can be before or after send)
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


@pytest.mark.single
@pytest.mark.skip("Need to update that test with src/dst")
def test_schedule():
	scheduler = SCHEDULERS["afab"]
	for placement in [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]]:
		for n_micro_batches in [2, 4, 8]:
			schedule = scheduler(placement, n_micro_batches)
			check_validity(schedule, placement, n_micro_batches)

	scheduler = SCHEDULERS["1f1b"]
	for placement in [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 2, 3]]:
		for n_micro_batches in [2, 4, 8]:
			schedule = scheduler(placement, n_micro_batches)
			check_validity(schedule, placement, n_micro_batches)

	scheduler = SCHEDULERS["hanayo"]
	for placement in [
		[0, 1, 2, 3, 3, 2, 1, 0],
		[0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
		[0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0],
	]:
		for n_micro_batches in [2, 4, 8]:
			schedule = scheduler(placement, n_micro_batches)
			check_validity(schedule, placement, n_micro_batches)
