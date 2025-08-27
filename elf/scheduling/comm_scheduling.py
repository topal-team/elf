"""
Static analysis of communications to avoid deadlocks, using a dependency graph representation of the schedule.
"""

import numpy as np
from typing import List, Tuple, Set, Dict
from collections import defaultdict

from .scheduling import OperationType, get_peer, matching, comm_types, compute_types, Operation


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
		self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)

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

		if not self.has_edge(source, target):
			self.adjacency_list[source].append(target)
			self.reverse_adjacency_list[target].append(source)
			self.adjacency_matrix[source, target] = True

	def remove_edge(self, source: int, target: int) -> None:
		"""
		Remove an edge from source to target.

		:param source: Source node
		:type source: int
		:param target: Target node
		:type target: int
		"""
		if self.has_edge(source, target):
			self.adjacency_list[source].remove(target)
			self.reverse_adjacency_list[target].remove(source)
			self.adjacency_matrix[source, target] = False

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
		return self.adjacency_matrix[source, target]

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


def smart_topological_sort(graph: DirectedGraph, schedule: List[Operation]) -> List[int]:
	"""
	Topological sort using simulation of execution.
	We use a step mechanism to try to synchronize communications when both send and recv are ready.
	"""
	node_to_op = {i: op for i, op in enumerate(schedule)}
	op_to_node = {op: i for i, op in enumerate(schedule)}
	out_degree = [0] * graph.num_nodes
	pairings = {}  # match send to corresponding recv

	for node in range(graph.num_nodes):
		out_degree[node] = graph.get_out_degree(node)

		for neighbor in graph.get_predecessors(node):
			if graph.has_bidirectional_edge(node, neighbor):
				pairings[node] = neighbor

	rank_ops = {}
	for op in schedule:
		if op.rank not in rank_ops:
			rank_ops[op.rank] = []
		rank_ops[op.rank].append(op)

	result = []

	def is_ready(node: int) -> bool:
		# Computes are ready when all dependencies are satisfied
		# Communications are ready when both send and recv are ready
		return out_degree[node] == 0 or (
			node in pairings and out_degree[node] <= 1 and out_degree[pairings[node]] <= 1
		)

	def process_node(node: int):
		# Mark a node as processed (added to the result), and update the dependency count of its predecessors
		result.append(node)
		for neighbor in graph.get_predecessors(node):
			out_degree[neighbor] -= 1

		out_degree[node] = -1

	# Pre-schedule all ops that don't have any dependencies (that's recv_forward on rank 0)
	to_process = [i for i, op in enumerate(schedule) if is_ready(i)]
	for i in to_process:
		process_node(i)

	step = 0
	while True:
		step += 1

		# Before each timestep, we schedule all communications that are ready
		# We add send and recv together in the results, to force them to be consecutive
		comms_to_process = []
		for rank, ops in rank_ops.items():
			ready_ops = [op for op in ops if is_ready(op_to_node[op])]
			send_ops = [
				op for op in ready_ops if op.op in {OperationType.SEND_FORWARD, OperationType.SEND_BACKWARD}
			]

			for send_op in send_ops:
				send_node = op_to_node[send_op]

				# Will be processed later
				if send_node not in pairings:
					continue

				comms_to_process.append(send_node)
				rank_ops[rank].remove(send_op)

				recv_node = pairings[send_node]
				recv_op = node_to_op[recv_node]
				rank_ops[recv_op.rank].remove(recv_op)
				comms_to_process.append(recv_node)

		for comm in comms_to_process:
			process_node(comm)

		# Then we do the actual timestep: find the next computation to be done, and execute it
		for rank, ops in rank_ops.items():
			ready_ops = [op for op in ops if is_ready(op_to_node[op])]

			# Special cases: comms with no peer (fake p2p), loss_forward, loss_backward and other stuff should be added and skipped in a loop
			# If we don't do this, the timestep will be desynchronized on different processors
			skippable_ops = [
				op for op in ready_ops if op.op not in compute_types and op_to_node[op] not in pairings
			]
			while len(skippable_ops) > 0:
				for op in skippable_ops:
					process_node(op_to_node[op])
					rank_ops[rank].remove(op)

				ready_ops = [op for op in ops if is_ready(op_to_node[op])]
				skippable_ops = [
					op for op in ready_ops if op.op not in compute_types and op_to_node[op] not in pairings
				]

			# And finally we do the computation
			for op in ready_ops:
				if op.op in compute_types:
					process_node(op_to_node[op])
					rank_ops[rank].remove(op)
					break

		# If we processed all ops, we're done
		if len(result) == len(schedule):
			break

	# Check that the schedule is valid (acyclic)
	# for i, op in enumerate(result):
	# 	for j in range(i + 1, len(result)):
	# 		assert not graph.has_unidirectional_edge(i, j), (
	# 			"Backward edge found in the toposort! The schedule with reordered communications can hang."
	# 		)

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
		# Count these 2 as computations to avoid moving them around
		OperationType.ALL_REDUCE_PARAM_GRADS,
		OperationType.PREFETCH_ACTIVATIONS,
	}

	# Create graph with one node per operation
	graph = DirectedGraph(len(schedule))

	# Track operations by rank and micro-batch for communication dependencies
	rank_mb_ops: Dict[Tuple[int, int], List[Tuple[int, Operation]]] = {}

	# First pass: collect operations by block_id and micro-batch
	for i, op in enumerate(schedule):
		mb_id = (
			op.mb_id if op.mb_id is not None else -1
		)  # use -1 for ALL_REDUCE operations and such, it won't be used anyway
		key = (op.block_id, mb_id)
		if key not in rank_mb_ops:
			rank_mb_ops[key] = []
		rank_mb_ops[key].append((i, op))

	def find_matching_comm(op):
		peer = get_peer(op)
		op_type = matching(op.op)
		key = (peer, op.mb_id)

		if key not in rank_mb_ops:
			return -1

		for i, other_op in rank_mb_ops[key]:
			if other_op.op == op_type and get_peer(other_op) == op.block_id:
				return i

		return -1

	def find_all_ops(block_id, mb_id, op_type):
		key = (block_id, mb_id)
		if key not in rank_mb_ops:
			return []

		return [i for i, op in rank_mb_ops[key] if op.op == op_type]

	# Second pass: add edges
	for i, op in enumerate(schedule):
		if op.op in comm_types:
			# Same micro-batch, peer block, matching operation
			j = find_matching_comm(op)
			if j != -1:
				graph.add_edge(i, j)
				graph.add_edge(j, i)

		if op.op in op_to_direct_dependency:
			# Same rank, same micro-batch, sequential dependency (rf->f->sf, rb->b->sb)
			# We assume that they are always before in the schedule for faster search
			deps = find_all_ops(op.block_id, op.mb_id, op_to_direct_dependency[op.op])
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

	return graph


def reorder_communications(schedule: List[Operation], strategy: str = "smart") -> List[Operation]:
	"""
	Reorder communications in the schedule to break all dependency cycles.

	:param schedule: List of operations to reorder
	:type schedule: List[Operation]
	:return: Reordered schedule
	:rtype: List[Operation]
	"""
	graph = schedule_to_graph(schedule)

	if strategy == "smart":
		topo_order = smart_topological_sort(graph, schedule)
	else:
		topo_order = topological_sort(graph)

	new_schedule = [schedule[i] for i in topo_order]
	return new_schedule
