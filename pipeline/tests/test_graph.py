import pytest
from ..graph import Operation, OperationType, graph_from_schedule


@pytest.mark.single
def test_graph_creation():
	# Simple example
	schedule = [
		Operation(0, 0, OperationType.SEND_FORWARD, 0),
		Operation(1, 0, OperationType.FORWARD, 0),
	]
	graph = graph_from_schedule(schedule)

	assert isinstance(graph, dict)
	g = graph[(0, 0)]
	assert isinstance(g, Operation)
	assert g.block_id == 1
	assert g.mb_id == 0
	assert g.op == OperationType.FORWARD
	assert len(g.dependencies) == 1  # forward depends on previous send_forward

	g = g.dependencies[0]
	assert g.block_id == 0
	assert g.mb_id == 0
	assert g.op == OperationType.SEND_FORWARD
	assert len(g.dependencies) == 0

	schedule = [
		Operation(0, 0, OperationType.RECV_FORWARD, 0),
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.SEND_FORWARD, 0),
		Operation(0, 0, OperationType.RECV_BACKWARD, 0),
		Operation(0, 0, OperationType.BACKWARD, 0),
		Operation(0, 0, OperationType.SEND_BACKWARD, 0),
	]
	graph = graph_from_schedule(schedule)

	g = graph[(0, 0)]
	assert g.block_id == 0
	assert g.mb_id == 0
	assert g.op == OperationType.SEND_BACKWARD
	assert len(g.dependencies) == 1  # SEND_FORWARD
