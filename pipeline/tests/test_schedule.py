from ..schedule import *
import pytest


@pytest.mark.single
def test_afab():
	placement = [0, 1]
	n_micro_batches = 2

	schedule = generate_afab_schedule(placement, n_micro_batches)

	assert schedule == [
		Operation(0, 0, OperationType.RECV_FORWARD, 0),
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.SEND_FORWARD, 0),
		Operation(0, 1, OperationType.RECV_FORWARD, 0),
		Operation(0, 1, OperationType.FORWARD, 0),
		Operation(0, 1, OperationType.SEND_FORWARD, 0),
		Operation(0, 0, OperationType.RECV_BACKWARD, 0),
		Operation(0, 0, OperationType.BACKWARD, 0),
		Operation(0, 0, OperationType.SEND_BACKWARD, 0),
		Operation(0, 1, OperationType.RECV_BACKWARD, 0),
		Operation(0, 1, OperationType.BACKWARD, 0),
		Operation(0, 1, OperationType.SEND_BACKWARD, 0),
		Operation(0, None, OperationType.ALL_REDUCE_PARAM_GRADS, 0),
		Operation(1, 0, OperationType.RECV_FORWARD, 1),
		Operation(1, 0, OperationType.FORWARD, 1),
		Operation(1, 0, OperationType.SEND_FORWARD, 1),
		Operation(1, 1, OperationType.RECV_FORWARD, 1),
		Operation(1, 1, OperationType.FORWARD, 1),
		Operation(1, 1, OperationType.SEND_FORWARD, 1),
		Operation(1, 0, OperationType.RECV_BACKWARD, 1),
		Operation(1, 0, OperationType.BACKWARD, 1),
		Operation(1, 0, OperationType.SEND_BACKWARD, 1),
		Operation(1, 1, OperationType.RECV_BACKWARD, 1),
		Operation(1, 1, OperationType.BACKWARD, 1),
		Operation(1, 1, OperationType.SEND_BACKWARD, 1),
		Operation(1, None, OperationType.ALL_REDUCE_PARAM_GRADS, 1),
	]


@pytest.mark.single
def test_1f1b():
	placement = [0, 1]
	n_micro_batches = 4

	schedule = generate_1f1b_schedule(placement, n_micro_batches)
	schedule0 = list(filter(lambda op: op.rank == 0, schedule))

	assert schedule0 == [
		Operation(0, 0, OperationType.RECV_FORWARD, 0),
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.SEND_FORWARD, 0),
		Operation(0, 1, OperationType.RECV_FORWARD, 0),
		Operation(0, 1, OperationType.FORWARD, 0),
		Operation(0, 1, OperationType.SEND_FORWARD, 0),
		Operation(0, 0, OperationType.RECV_BACKWARD, 0),
		Operation(0, 0, OperationType.BACKWARD, 0),
		Operation(0, 0, OperationType.SEND_BACKWARD, 0),
		Operation(0, 2, OperationType.RECV_FORWARD, 0),
		Operation(0, 2, OperationType.FORWARD, 0),
		Operation(0, 2, OperationType.SEND_FORWARD, 0),
		Operation(0, 1, OperationType.RECV_BACKWARD, 0),
		Operation(0, 1, OperationType.BACKWARD, 0),
		Operation(0, 1, OperationType.SEND_BACKWARD, 0),
		Operation(0, 3, OperationType.RECV_FORWARD, 0),
		Operation(0, 3, OperationType.FORWARD, 0),
		Operation(0, 3, OperationType.SEND_FORWARD, 0),
		Operation(0, 2, OperationType.RECV_BACKWARD, 0),
		Operation(0, 2, OperationType.BACKWARD, 0),
		Operation(0, 2, OperationType.SEND_BACKWARD, 0),
		Operation(0, 3, OperationType.RECV_BACKWARD, 0),
		Operation(0, 3, OperationType.BACKWARD, 0),
		Operation(0, 3, OperationType.SEND_BACKWARD, 0),
		Operation(0, None, OperationType.ALL_REDUCE_PARAM_GRADS, 0),
	]

	schedule1 = list(filter(lambda op: op.rank == 1, schedule))

	# Send/Recv are swapped on odd devices to avoid deadlocks :)
	assert schedule1 == [
		Operation(1, 0, OperationType.RECV_FORWARD, 1),
		Operation(1, 0, OperationType.FORWARD, 1),
		Operation(1, 0, OperationType.SEND_FORWARD, 1),
		Operation(1, 0, OperationType.RECV_BACKWARD, 1),
		Operation(1, 0, OperationType.BACKWARD, 1),
		Operation(1, 0, OperationType.SEND_BACKWARD, 1),
		Operation(1, 1, OperationType.RECV_FORWARD, 1),
		Operation(1, 1, OperationType.FORWARD, 1),
		Operation(1, 1, OperationType.SEND_FORWARD, 1),
		Operation(1, 1, OperationType.RECV_BACKWARD, 1),
		Operation(1, 1, OperationType.BACKWARD, 1),
		Operation(1, 1, OperationType.SEND_BACKWARD, 1),
		Operation(1, 2, OperationType.RECV_FORWARD, 1),
		Operation(1, 2, OperationType.FORWARD, 1),
		Operation(1, 2, OperationType.SEND_FORWARD, 1),
		Operation(1, 2, OperationType.RECV_BACKWARD, 1),
		Operation(1, 2, OperationType.BACKWARD, 1),
		Operation(1, 2, OperationType.SEND_BACKWARD, 1),
		Operation(1, 3, OperationType.RECV_FORWARD, 1),
		Operation(1, 3, OperationType.FORWARD, 1),
		Operation(1, 3, OperationType.SEND_FORWARD, 1),
		Operation(1, 3, OperationType.RECV_BACKWARD, 1),
		Operation(1, 3, OperationType.BACKWARD, 1),
		Operation(1, 3, OperationType.SEND_BACKWARD, 1),
		Operation(1, None, OperationType.ALL_REDUCE_PARAM_GRADS, 1),
	]
