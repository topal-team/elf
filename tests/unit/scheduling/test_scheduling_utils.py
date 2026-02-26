import pytest

from elf.scheduling import schedule_to_str
from elf.scheduling.scheduling import matching, get_peer, get_peer_rank, OperationType, Operation


@pytest.mark.unit
def test_matching():
	assert matching(OperationType.RECV_FORWARD) == OperationType.SEND_FORWARD
	assert matching(OperationType.SEND_FORWARD) == OperationType.RECV_FORWARD
	assert matching(OperationType.RECV_BACKWARD) == OperationType.SEND_BACKWARD
	assert matching(OperationType.SEND_BACKWARD) == OperationType.RECV_BACKWARD


@pytest.mark.unit
def test_get_peer():
	op_recv = Operation(0, 0, OperationType.RECV_FORWARD, 0, src=1)
	assert get_peer(op_recv) == 1

	op_send = Operation(1, 0, OperationType.SEND_FORWARD, 1, dst=0)
	assert get_peer(op_send) == 0

	op_recv_back = Operation(2, 1, OperationType.RECV_BACKWARD, 2, src=3)
	assert get_peer(op_recv_back) == 3

	op_send_back = Operation(3, 1, OperationType.SEND_BACKWARD, 3, dst=2)
	assert get_peer(op_send_back) == 2


@pytest.mark.unit
def test_get_peer_rank():
	placement = [0, 1, 2, 3]

	op_recv = Operation(0, 0, OperationType.RECV_FORWARD, 0, src=1)
	assert get_peer_rank(op_recv, placement) == 1

	op_send = Operation(2, 0, OperationType.SEND_FORWARD, 2, dst=3)
	assert get_peer_rank(op_send, placement) == 3

	op_recv_back = Operation(3, 1, OperationType.RECV_BACKWARD, 3, src=2)
	assert get_peer_rank(op_recv_back, placement) == 2


@pytest.mark.unit
def test_get_peer_rank_interleaved():
	placement = [0, 1, 2, 3, 0, 1, 2, 3]

	op_recv = Operation(4, 0, OperationType.RECV_FORWARD, 4, src=3)
	assert get_peer_rank(op_recv, placement) == 3

	op_send = Operation(3, 0, OperationType.SEND_FORWARD, 3, dst=4)
	assert get_peer_rank(op_send, placement) == 0


@pytest.mark.unit
def test_schedule_to_str():
	schedule = [
		Operation(0, 0, OperationType.RECV_FORWARD, 0, src=None),
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.SEND_FORWARD, 0, dst=1),
		Operation(1, 0, OperationType.RECV_FORWARD, 1, src=0),
		Operation(1, 0, OperationType.FORWARD, 1),
		Operation(1, 0, OperationType.SEND_FORWARD, 1, dst=None),
	]

	schedule_str = schedule_to_str(schedule, print_comms=False)

	assert isinstance(schedule_str, str)
	assert len(schedule_str) > 0


@pytest.mark.unit
def test_operation_equality():
	op1 = Operation(0, 0, OperationType.FORWARD, 0)
	op2 = Operation(0, 0, OperationType.FORWARD, 0)
	op3 = Operation(0, 1, OperationType.FORWARD, 0)

	assert op1 == op2
	assert op1 != op3
