import pytest
from ..schedule import Operation, Operations, graph_from_schedule

@pytest.mark.single
def test_graph_creation():
    # Simple example
    schedule = [Operation(0, 0, Operations.FORWARD, 0), Operation(0, 0, Operations.BACKWARD, 0)]
    graph = graph_from_schedule(schedule)

    assert isinstance(graph, dict)
    g = graph[(0, 0)]
    assert isinstance(g, Operation)
    assert g.block_id == 0
    assert g.mb_id == 0
    assert g.op == Operations.BACKWARD
    assert len(g.dependencies) == 1

    g = g.dependencies[0]
    assert g.block_id == 0
    assert g.mb_id == 0
    assert g.op == Operations.FORWARD
    assert len(g.dependencies) == 0

    schedule = [Operation(0, 0, Operations.RECV_FORWARD, 0), Operation(0, 0, Operations.FORWARD, 0), Operation(0, 0, Operations.SEND_FORWARD, 0), Operation(0, 0, Operations.RECV_BACKWARD, 0), Operation(0, 0, Operations.BACKWARD, 0), Operation(0, 0, Operations.SEND_BACKWARD, 0)]
    graph = graph_from_schedule(schedule)

    g = graph[(0, 0)]
    assert g.block_id == 0
    assert g.mb_id == 0
    assert g.op == Operations.SEND_BACKWARD
    assert len(g.dependencies) == 2 # RECV_BACKWARD, BACKWARD
    uniq = {}
    stack = [g]
    while stack:
        if g not in uniq:
            uniq[g] = True
            for d in g.dependencies:
                stack.append(d)
        g = stack.pop(0)
    assert len(uniq.keys()) == 6