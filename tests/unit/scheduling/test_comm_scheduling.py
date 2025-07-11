import pytest

from elf.scheduling import Operation, OperationType
from elf.scheduling.comm_scheduling import (
	DirectedGraph,
	topological_sort,
	schedule_to_graph,
	reorder_communications,
)


def _is_topologically_sorted(graph: DirectedGraph, order: list[int]) -> bool:
	"""
	Topological order check, with the exception that bidirectional edges are ignored.
	"""
	pos = {node: idx for idx, node in enumerate(order)}
	for src in range(graph.num_nodes):
		for dst in graph.get_successors(src):
			# src depends on dst → dst must come before src
			# except if dst is bidirectional with src
			if pos[src] <= pos[dst] and not graph.has_bidirectional_edge(dst, src):
				return False

	return True


@pytest.mark.unit
def test_directed_graph_basic():
	"""Basic CRUD operations on DirectedGraph."""
	g = DirectedGraph(3)
	g.add_edge(0, 1)
	g.add_edge(1, 2)

	# Edges / degrees
	assert g.has_edge(0, 1)
	assert g.has_edge(1, 2)
	assert not g.has_edge(0, 2)
	assert g.get_out_degree(0) == 1
	assert g.get_in_degree(1) == 1

	# Successors / predecessors
	assert g.get_successors(0) == [1]
	assert g.get_predecessors(2) == [1]

	# Removing an edge updates structures consistently
	g.remove_edge(1, 2)
	assert not g.has_edge(1, 2)
	assert g.get_out_degree(1) == 0
	assert g.get_in_degree(2) == 0


@pytest.mark.unit
def test_cycle_detection():
	"""`has_cycle` must ignore 2-cycles but detect longer ones."""
	# Bidirectional (2-cycle) should be ignored
	g1 = DirectedGraph(2)
	g1.add_edge(0, 1)
	g1.add_edge(1, 0)
	assert not g1.has_cycle()

	# Length-3 cycle must be reported
	g2 = DirectedGraph(3)
	g2.add_edge(0, 1)
	g2.add_edge(1, 2)
	g2.add_edge(2, 0)
	assert g2.has_cycle()


@pytest.mark.unit
def test_topological_sort_with_bidirectional_edges():
	"""`topological_sort` should respect dependencies and keep bidirectional nodes consecutive."""
	g = DirectedGraph(3)
	# Bidirectional edge between 0 and 1
	g.add_edge(0, 1)
	g.add_edge(1, 0)
	# Extra dependency: 1 depends on 2
	g.add_edge(1, 2)

	order = topological_sort(g)

	# Node 2 has no outgoing edges – must be first
	assert order[0] == 2

	# Nodes 0 and 1 must be consecutive (any order)
	idx0, idx1 = order.index(0), order.index(1)
	assert abs(idx0 - idx1) == 1

	# Validate ordering w.r.t. dependencies
	assert _is_topologically_sorted(g, order)


@pytest.mark.unit
def test_reorder_communications_simple():
	"""`reorder_communications` should reorder ops to satisfy dependencies."""
	# Deliberately unsorted schedule: forward precedes its recv dependency
	schedule = [
		Operation(0, 0, OperationType.FORWARD, 0),
		Operation(0, 0, OperationType.RECV_FORWARD, 0, src=None),
	]

	new_schedule = reorder_communications(schedule)

	# Expected order: recv → forward
	assert [op.op for op in new_schedule] == [OperationType.RECV_FORWARD, OperationType.FORWARD]

	# The reordered schedule must be (reverse) topologically sorted
	graph = schedule_to_graph(schedule)
	original_to_index = {
		idx: position for position, idx in enumerate([schedule.index(op) for op in new_schedule])
	}
	for src in range(graph.num_nodes):
		for dst in graph.get_successors(src):
			# src depends on dst → dst appears before src in the new order
			assert original_to_index[src] > original_to_index[dst]
