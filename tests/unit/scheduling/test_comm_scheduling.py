import pytest

from elf.scheduling.comm_scheduling import DirectedGraph


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
