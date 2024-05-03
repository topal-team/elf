import pytest
from ..schedule import Operation, Operations, graph_from_schedule

@pytest.mark.single
def test_graph_creation():
    # Simple example
    schedule = [Operation(0, 0, Operations.FORWARD), Operation(0, 0, Operations.BACKWARD)]
    graph = graph_from_schedule(schedule)

    assert isinstance(graph, Operation)
    assert graph.block_id == 0
    assert graph.mb_id == 0
    assert graph.op == Operations.BACKWARD
    assert len(graph.dependencies) == 1

    graph = graph.dependencies[0]
    assert graph.block_id == 0
    assert graph.mb_id == 0
    assert graph.op == Operations.FORWARD
    assert len(graph.dependencies) == 0

    schedule = [Operation(0, 0, Operations.RECV_FORWARD), Operation(0, 0, Operations.FORWARD), Operation(0, 0, Operations.SEND_FORWARD), Operation(0, 0, Operations.RECV_BACKWARD), Operation(0, 0, Operations.BACKWARD), Operation(0, 0, Operations.SEND_BACKWARD)]
    graph = graph_from_schedule(schedule)

    assert graph.block_id == 0
    assert graph.mb_id == 0
    assert graph.op == Operations.SEND_BACKWARD
    print(graph.dependencies)
    assert len(graph.dependencies) == 2 # RECV_BACKWARD, BACKWARD
    uniq = {}
    stack = [graph]
    while stack:
        if graph not in uniq:
            uniq[graph] = True
            for d in graph.dependencies:
                stack.append(d)
        graph = stack.pop(0)
    assert len(uniq.keys()) == 6