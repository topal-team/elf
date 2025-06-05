"""
Static analysis of communications to avoid deadlocks, using a dependency graph representation of the schedule.
"""

from typing import List, Tuple, Set, Dict
from collections import defaultdict

from .scheduling import OperationType, get_peer, matching, comm_types, Operation


class DirectedGraph:
	"""
	A simple directed graph structure.
	"""

	def __init__(self, num_nodes: int):
		"""
		Initialize a directed graph with the specified number of nodes.

		:param num_nodes: Number of nodes in the graph
		:type num_nodes: int
		"""
		if num_nodes <= 0:
			raise ValueError("Number of nodes must be positive")

		self.num_nodes = num_nodes
		self.adjacency_list: Dict[int, List[int]] = defaultdict(list)
		self.reverse_adjacency_list: Dict[int, List[int]] = defaultdict(list)
		self.edges: Set[Tuple[int, int]] = set()

	def add_edge(self, source: int, target: int) -> None:
		"""
		Add an edge from source to target.

		:param source: Source node
		:type source: int
		:param target: Target node
		:type target: int
		"""
		if not (0 <= source < self.num_nodes and 0 <= target < self.num_nodes):
			raise ValueError("Node indices must be within valid range")

		if source == target:
			raise ValueError("Self-loops are not allowed")

		if (source, target) not in self.edges:
			self.edges.add((source, target))
			self.adjacency_list[source].append(target)
			self.reverse_adjacency_list[target].append(source)

	def remove_edge(self, source: int, target: int) -> None:
		"""
		Remove an edge from source to target.

		:param source: Source node
		:type source: int
		:param target: Target node
		:type target: int
		"""
		if (source, target) in self.edges:
			self.edges.remove((source, target))
			self.adjacency_list[source].remove(target)
			self.reverse_adjacency_list[target].remove(source)

	def get_successors(self, node: int) -> List[int]:
		"""
		Get all successors of a node.

		:param node: Node to get successors for
		:type node: int
		:return: List of successor nodes
		:rtype: List[int]
		"""
		return self.adjacency_list[node].copy()

	def get_predecessors(self, node: int) -> List[int]:
		"""
		Get all predecessors of a node.

		:param node: Node to get predecessors for
		:type node: int
		:return: List of predecessor nodes
		:rtype: List[int]
		"""
		return self.reverse_adjacency_list[node].copy()

	def has_edge(self, source: int, target: int) -> bool:
		"""
		Check if an edge exists from source to target.

		:param source: Source node
		:type source: int
		:param target: Target node
		:type target: int
		:return: True if edge exists, False otherwise
		:rtype: bool
		"""
		return (source, target) in self.edges

	def has_unidirectional_edge(self, node1: int, node2: int) -> bool:
		"""
		Check if there is a unidirectional edge between two nodes.
		"""
		return self.has_edge(node1, node2) and not self.has_edge(node2, node1)

	def has_bidirectional_edge(self, node1: int, node2: int) -> bool:
		"""
		Check if there is a bidirectional edge between two nodes.
		"""
		return self.has_edge(node1, node2) and self.has_edge(node2, node1)

	def get_in_degree(self, node: int) -> int:
		"""
		Get the in-degree of a node.

		:param node: Node to get in-degree for
		:type node: int
		:return: In-degree of the node
		:rtype: int
		"""
		return len(self.reverse_adjacency_list[node])

	def get_out_degree(self, node: int) -> int:
		"""
		Get the out-degree of a node.

		:param node: Node to get out-degree for
		:type node: int
		:return: Out-degree of the node
		:rtype: int
		"""
		return len(self.adjacency_list[node])

	def get_all_edges(self) -> List[Tuple[int, int]]:
		"""
		Get all edges in the graph.

		:return: List of edges as (source, target) tuples
		:rtype: List[Tuple[int, int]]
		"""
		return list(self.edges)

	def has_cycle(self) -> bool:
		"""
		Check if the graph has a cycle using DFS, ignoring 2-cycles.

		:return: True if graph has a cycle of length > 2, False otherwise
		:rtype: bool
		"""
		WHITE, GRAY, BLACK = 0, 1, 2
		color = [WHITE] * self.num_nodes

		def dfs(node: int, parent: int = -1) -> bool:
			if color[node] == GRAY:
				return True  # Back edge found, cycle detected
			if color[node] == BLACK:
				return False  # Already processed

			color[node] = GRAY
			for successor in self.get_successors(node):
				# Skip 2-cycles: if successor is parent and there's a back edge, it's a 2-cycle
				if successor == parent and self.has_edge(successor, node):
					continue

				if dfs(successor, node):
					return True
			color[node] = BLACK
			return False

		for i in range(self.num_nodes):
			if color[i] == WHITE:
				if dfs(i):
					return True

		return False


def topological_sort(graph: DirectedGraph) -> List[int]:
	"""
	Sort the nodes of the graph topologically (in reverse order).
	Nodes that are connected by bidirectional edges are forced to be consecutive in the output.
	"""

	out_degree = [0] * graph.num_nodes
	pairings = {}  # Nodes connected by bidirectional edges
	for node in range(graph.num_nodes):
		out_degree[node] = graph.get_out_degree(node)

		for neighbor in graph.get_predecessors(node):
			if graph.has_bidirectional_edge(node, neighbor):
				pairings[node] = neighbor

	def is_ready(node: int) -> int:
		degree = out_degree[node]
		if degree == 0:  # All dependencies are satisfied, this node is ready
			return True

		if degree != 1:  # More than one dependency, this node is not ready
			return False

		# If the only dependency is a bidirectional edge, this node is ready if the other is ready too
		if node in pairings:
			return out_degree[pairings[node]] == 1

		return False

	queue = []
	for node in range(graph.num_nodes):
		if is_ready(node):
			queue.append(node)

	result = []

	def process_node(node: int):
		result.append(node)

		for neighbor in graph.get_predecessors(node):
			out_degree[neighbor] -= 1
			if is_ready(neighbor):
				queue.append(neighbor)

		out_degree[node] = -1  # This node is now processed, so it should not be processed again

	while queue:
		node = queue.pop(0)
		if out_degree[node] == -1:
			continue

		process_node(node)

		if node in pairings:
			process_node(pairings[node])

	assert len(result) == graph.num_nodes, (
		f"Not all nodes were visited ({len(result)}/{graph.num_nodes})"
	)

	return result


def schedule_to_graph(schedule: List[Operation]) -> Tuple[DirectedGraph, Dict[int, Operation]]:
	"""
	Convert a schedule into a directed graph representation.

	:param schedule: List of operations to convert
	:type schedule: List[Operation]
	:return: Tuple of (directed graph representing the schedule dependencies, mapping from node ID to operation)
	:rtype: Tuple[DirectedGraph, Dict[int, Operation]]
	"""
	op_to_direct_dependency = {
		OperationType.FORWARD: OperationType.RECV_FORWARD,
		OperationType.SEND_FORWARD: OperationType.FORWARD,
		OperationType.BACKWARD_INPUTS: OperationType.RECV_BACKWARD,
		OperationType.SEND_BACKWARD: OperationType.BACKWARD_INPUTS,
		OperationType.RECV_BACKWARD: OperationType.LOSS_BACKWARD,
	}

	compute_ops = {
		OperationType.FORWARD,
		OperationType.BACKWARD_INPUTS,
		OperationType.BACKWARD_PARAMS,
		OperationType.LOSS_FORWARD,
		OperationType.LOSS_BACKWARD,
		OperationType.RECOMPUTE_FORWARD,
		OperationType.RECOMPUTE_BACKWARD_INPUTS,
		OperationType.ALL_REDUCE_PARAM_GRADS,
	}

	# Create graph with one node per operation
	graph = DirectedGraph(len(schedule))

	# Create mapping from node ID to operation
	node_to_op = {i: op for i, op in enumerate(schedule)}

	# Track operations by rank and micro-batch for communication dependencies
	rank_mb_ops: Dict[Tuple[int, int], List[Tuple[int, Operation]]] = {}

	# First pass: collect operations by rank
	for i, op in enumerate(schedule):
		mb_id = op.mb_id or -1  # use -1 for ALL_REDUCE operations and such, it won't be used anyway
		key = (op.rank, mb_id)
		if key not in rank_mb_ops:
			rank_mb_ops[key] = []
		rank_mb_ops[key].append((i, op))

	def find_op(condition):
		for i, op in enumerate(schedule):
			if condition(op):
				return i
		return -1

	def find_all_ops(condition):
		return [i for i, op in enumerate(schedule) if condition(op)]

	# Second pass: add edges
	for i, op in enumerate(schedule):
		if op.op in comm_types:
			# Same micro-batch, peer block, matching operation
			j = find_op(
				lambda matching_op: get_peer(op) == matching_op.block_id
				and op.block_id == get_peer(matching_op)
				and matching_op.mb_id == op.mb_id
				and matching(op.op) == matching_op.op
			)
			if j != -1:
				graph.add_edge(i, j)
				graph.add_edge(j, i)

		if op.op in op_to_direct_dependency:
			# Same rank, same micro-batch, sequential dependency (rf->f->sf, rb->b->sb)
			deps = find_all_ops(
				lambda matching_op: matching_op.block_id == op.block_id
				and matching_op.mb_id == op.mb_id
				and op_to_direct_dependency[op.op] == matching_op.op
			)
			for j in deps:
				graph.add_edge(i, j)

	# Add chain dependencies: sequential operations on same processor
	rank_ops: Dict[int, List[int]] = {}

	# Group operations by rank
	for i, op in enumerate(schedule):
		if op.rank not in rank_ops:
			rank_ops[op.rank] = []
		if op.op in compute_ops:
			rank_ops[op.rank].append(i)

	# Add edges between consecutive operations on the same rank
	for _, ops in rank_ops.items():
		for j in range(len(ops)):
			current_op_idx = ops[j]
			if j < len(ops) - 1:
				next_op_idx = ops[j + 1]
				graph.add_edge(next_op_idx, current_op_idx)

	return graph, node_to_op


def reorder_communications(schedule: List[Operation]) -> List[Operation]:
	"""
	Reorder communications in the schedule to break all dependency cycles.

	:param schedule: List of operations to reorder
	:type schedule: List[Operation]
	:return: Reordered schedule
	:rtype: List[Operation]
	"""
	graph, node_to_op = schedule_to_graph(schedule)
	topo_order = topological_sort(graph)
	new_schedule = [node_to_op[i] for i in topo_order]
	return new_schedule
