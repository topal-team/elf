"""
Static analysis of communications to avoid deadlocks, using a dependency graph representation of the schedule.
"""

import heapq
import math
from typing import List, Tuple, Set, Dict
from collections import defaultdict, deque
from bisect import bisect_left, bisect_right, insort

from elf.scheduling.scheduling import OperationType, matching, comm_types, compute_types, Operation
from elf.profiling import StageProfile
from elf.signature import Signature


import logging

logger = logging.getLogger(__name__)

__all__ = [
	"add_comms",
	"node_index",
	"by_rank",
	"chains",
	"matching_comm",
	"find_ops",
	"DirectedGraph",
	"schedule_to_graph",
	"recv_types",
	"send_types",
	"topological_sort",
	"smart_topological_sort",
	"pipelined_topological_sort",
	"overlap_comm_scheduler",
	"safe_overlap_comm_scheduler",
	"timed_overlap_comm_scheduler",
	"toposort_with_recv_prio_comm_scheduler",
]


recv_types = {OperationType.RECV_FORWARD, OperationType.RECV_BACKWARD}

send_types = {OperationType.SEND_FORWARD, OperationType.SEND_BACKWARD}

#: Relative slowdown of a computation while one NCCL p2p kernel of its device is in flight next to
#: it. A posted p2p kernel occupies a few SMs from the moment it starts, whether it is moving data or
#: still waiting (spinning) for its peer; each kernel in flight adds as much again. This is the
#: overlap cost of ``local/simulator_with_comms.py``, calibrated against measured iterations on H100
#: with NCCL's default number of CTAs.
P2P_SLOWDOWN = 0.1

#: How much a computation starting later, because its RECV was posted after its SEND, counts
#: against the waiting that saves in :py:func:`_just_in_time_targets`. 1 treats every delay as
#: lost time; a rank with slack absorbs part of it.
CONSUMER_DELAY_WEIGHT = 1.0


def node_index(schedule: List[Operation]) -> Dict[Operation, int]:
	"""
	Position of every operation in the schedule, to go from an operation back to its node.

	:param schedule: the schedule to index
	:type schedule: List[Operation]
	:rtype: Dict[Operation, int]
	"""
	return {op: node for node, op in enumerate(schedule)}


def by_rank(schedule: List[Operation]) -> Dict[int, List[Operation]]:
	"""
	Operations of the schedule, grouped by the rank executing them, in schedule order.

	:param schedule: the schedule to group
	:type schedule: List[Operation]
	:rtype: Dict[int, List[Operation]]
	"""
	grouped = defaultdict(list)
	for op in schedule:
		grouped[op.rank].append(op)
	return grouped


def chains(schedule: List[Operation]) -> Dict[Tuple[int, int], List[int]]:
	"""
	Nodes of the schedule, grouped by the (block, micro-batch) they belong to.

	Everything an operation depends on locally -- the receive it waits for, the computation it
	sends the result of, the communication it rendezvous with on its peer -- is in one of those
	groups, which is what makes them worth indexing once.

	:param schedule: the schedule to index
	:type schedule: List[Operation]
	:rtype: Dict[Tuple[int, int], List[int]]
	"""
	grouped = defaultdict(list)
	for node, op in enumerate(schedule):
		grouped[op.block_id, op.mb_id].append(node)
	return grouped


def matching_comm(schedule: List[Operation], groups: Dict, op: Operation) -> int | None:
	"""
	Node of the communication *op* rendezvous with: the opposite one, on the peer block.

	:param schedule: the schedule *groups* indexes
	:type schedule: List[Operation]
	:param groups: the (block, micro-batch) groups, from :func:`chains`
	:type groups: Dict[Tuple[int, int], List[int]]
	:param op: the communication to match
	:type op: Operation

	:return: the node of the matching communication, or None when it has no peer
	:rtype: int or None
	"""
	for node in groups.get((op.peer, op.mb_id), ()):
		other = schedule[node]
		if other.op == matching(op.op) and other.peer == op.block_id:
			return node
	return None


def find_ops(
	schedule: List[Operation], groups: Dict, block_id: int, mb_id: int, op_type: OperationType
) -> List[int]:
	"""
	Nodes of every *op_type* operation of a (block, micro-batch).

	:param schedule: the schedule *groups* indexes
	:type schedule: List[Operation]
	:param groups: the (block, micro-batch) groups, from :func:`chains`
	:type groups: Dict[Tuple[int, int], List[int]]

	:rtype: List[int]
	"""
	return [node for node in groups.get((block_id, mb_id), ()) if schedule[node].op == op_type]


class DirectedGraph:
	"""
	A simple directed graph structure.

	Schedules make very sparse graphs -- an operation depends on the one before it on its rank and
	on the communication it waits for, so a few edges per node whatever the size of the schedule --
	so edges are stored as a set of pairs rather than as a matrix.
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

		if not self.has_edge(source, target):
			self.adjacency_list[source].append(target)
			self.reverse_adjacency_list[target].append(source)
			self.edges.add((source, target))

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
			self.edges.discard((source, target))

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

	def has_cycle(self) -> bool:
		"""
		Check if the graph has a cycle using DFS, ignoring 2-cycles.

		:return: True if graph has a cycle of length > 2, False otherwise
		:rtype: bool
		"""
		WHITE, GRAY, BLACK = 0, 1, 2
		color = [WHITE] * self.num_nodes

		# Depth-first, iteratively: a schedule has one node per operation, which is far too many to
		# walk with the call stack. Each frame keeps the node it explores, the node it was reached
		# from, and where it got to in its successors.
		for start in range(self.num_nodes):
			if color[start] != WHITE:
				continue

			color[start] = GRAY
			stack = [(start, -1, iter(self.get_successors(start)))]
			while stack:
				node, parent, successors = stack[-1]

				descended = False
				for successor in successors:
					# Skip 2-cycles: if successor is parent and there's a back edge, it's a 2-cycle
					if successor == parent and self.has_edge(successor, node):
						continue

					if color[successor] == GRAY:
						return True  # Back edge found, cycle detected
					if color[successor] == WHITE:
						color[successor] = GRAY
						stack.append((successor, node, iter(self.get_successors(successor))))
						descended = True
						break

				if not descended:  # every successor explored
					color[node] = BLACK
					stack.pop()

		return False

	def has_path(self, i: int, j: int):
		"""
		Find a path from *i* to *j*, following the direction of the edges.

		:return: the nodes of a path, starting with *i* and ending with *j*, or False if there is
			none. Any path will do, it is not the shortest one.
		:rtype: List[int] or bool
		"""
		# Depth-first, iteratively, remembering how each node was reached. Visited nodes are never
		# explored twice, which also breaks the 2-cycles a matched communication makes.
		reached_from = {i: None}
		stack = [i]
		while stack:
			node = stack.pop()
			if node == j:
				path = []
				while node is not None:
					path.append(node)
					node = reached_from[node]
				return list(reversed(path))

			for successor in reversed(self.get_successors(node)):
				if successor not in reached_from:
					reached_from[successor] = node
					stack.append(successor)

		return False


def topological_sort(
	schedule: List[Operation], profiles: List[StageProfile] | None = None
) -> List[Operation]:
	"""
	Sort the nodes of the graph topologically (in reverse order).
	Nodes that are connected by bidirectional edges are forced to be consecutive in the output.
	"""
	graph = schedule_to_graph(schedule)
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

	queue = deque()
	for node in range(graph.num_nodes):
		if is_ready(node):
			queue.append(node)

	result = []

	def process_node(node: int):
		result.append(schedule[node])

		for neighbor in graph.get_predecessors(node):
			out_degree[neighbor] -= 1
			if is_ready(neighbor):
				queue.append(neighbor)

		out_degree[node] = -1  # This node is now processed, so it should not be processed again

	while queue:
		node = queue.popleft()
		if out_degree[node] == -1:
			continue

		process_node(node)

		if node in pairings:
			process_node(pairings[node])

	assert len(result) == graph.num_nodes, (
		f"Not all nodes were visited ({len(result)}/{graph.num_nodes})"
	)

	return result


def smart_topological_sort(
	schedule: List[Operation], profiles: List[StageProfile] | None = None
) -> List[Operation]:
	"""
	Topological sort using simulation of execution.
	We use a step mechanism to try to synchronize communications when both send and recv are ready.
	"""
	graph = schedule_to_graph(schedule)
	op_to_node = node_index(schedule)
	out_degree = [0] * graph.num_nodes
	pairings = {}  # match send to corresponding recv

	for node in range(graph.num_nodes):
		out_degree[node] = graph.get_out_degree(node)

		for neighbor in graph.get_predecessors(node):
			if graph.has_bidirectional_edge(node, neighbor):
				pairings[node] = neighbor

	rank_ops = by_rank(schedule)

	result = []

	def is_ready(node: int) -> bool:
		# Computes are ready when all dependencies are satisfied
		# Communications are ready when both send and recv are ready
		return out_degree[node] == 0 or (
			node in pairings and out_degree[node] <= 1 and out_degree[pairings[node]] <= 1
		)

	def process_node(node: int):
		# Mark a node as processed (added to the result), and update the dependency count of its predecessors
		result.append(schedule[node])
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
				recv_op = schedule[recv_node]
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

	return result


def pipelined_topological_sort(
	schedule: List[Operation], profiles: List[StageProfile] | None = None
) -> List[Operation]:
	"""
	Topological sort using pipelined execution.
	"""
	graph = schedule_to_graph(schedule)
	op_to_node = node_index(schedule)
	out_degree = [0] * graph.num_nodes
	pairings = {}  # match send to corresponding recv

	for node in range(graph.num_nodes):
		out_degree[node] = graph.get_out_degree(node)

		for neighbor in graph.get_predecessors(node):
			if graph.has_bidirectional_edge(node, neighbor):
				pairings[node] = neighbor

	rank_ops = by_rank(schedule)

	result = []

	def is_ready(node: int) -> bool:
		# Computes are ready when all dependencies are satisfied
		# Communications are ready when both send and recv are ready
		return out_degree[node] == 0 or (
			node in pairings and out_degree[node] <= 1 and out_degree[pairings[node]] <= 1
		)

	def process_node(node: int):
		# Mark a node as processed (added to the result), and update the dependency count of its predecessors
		for neighbor in graph.get_predecessors(node):
			out_degree[neighbor] -= 1

		out_degree[node] = -1

	# Pre-schedule all ops that don't have any dependencies (that's recv_forward on rank 0)
	# to_process = [i for i, op in enumerate(schedule) if is_ready(i)]
	# for i in to_process:
	# 	process_node(i)

	while True:
		to_process = {rank: [] for rank in rank_ops.keys()}
		for rank, ops in rank_ops.items():
			ready_ops = [op for op in ops if is_ready(op_to_node[op])]

			# Special cases: comms with no peer (fake p2p), loss_forward, loss_backward and other stuff should be added and skipped in a loop
			# If we don't do this, the timestep will be desynchronized on different processors
			skippable_ops = [
				op for op in ready_ops if op.op not in compute_types and op.op not in comm_types
			]
			while len(skippable_ops) > 0:
				for op in skippable_ops:
					result.append(op)  # append and process immediately
					process_node(op_to_node[op])
					rank_ops[rank].remove(op)

				ready_ops = [op for op in ops if is_ready(op_to_node[op])]
				skippable_ops = [
					op for op in ready_ops if op.op not in compute_types and op_to_node[op] not in pairings
				]

			next_recv = next((op for op in ops if op.op in recv_types and is_ready(op_to_node[op])), None)
			if next_recv is not None:
				result.append(
					next_recv
				)  # append but don't process yet ; it will be finished after this step
				to_process[rank].append(op_to_node[next_recv])
				rank_ops[rank].remove(next_recv)

			next_compute = next(
				(op for op in ops if op.op in compute_types and is_ready(op_to_node[op])), None
			)
			if next_compute is not None:
				rank_ops[rank].remove(next_compute)
				result.append(next_compute)  # append AND process immediately so that the next send is ready
				process_node(op_to_node[next_compute])

			next_send = next((op for op in ops if op.op in send_types and is_ready(op_to_node[op])), None)
			if next_send is not None:
				result.append(next_send)  # append but don't process yet ; it will be ready after this step
				to_process[rank].append(op_to_node[next_send])
				rank_ops[rank].remove(next_send)

		for rank, nodes in to_process.items():
			for node in nodes:
				process_node(node)

			nodes.clear()

		if len(result) == len(schedule):
			break

	for i, op in enumerate(result):
		for j in range(i + 1, len(result)):
			assert not graph.has_unidirectional_edge(op_to_node[result[i]], op_to_node[result[j]]), (
				"Backward edge found in the toposort! The schedule with reordered communications can hang."
				+ f" {result[i]} -> {result[j]}"
			)
			assert j == i + 1 or not graph.has_bidirectional_edge(
				op_to_node[result[i]], op_to_node[result[j]]
			), f"Two communications are not placed next to each other: {result[i]} <-> {result[j]}"

	return result


def overlap_comm_scheduler(
	schedule: List[Operation], profiles: List[StageProfile] | None = None, prefetch=1
) -> List[Operation]:
	"""
	With NCCL_MAX_CTAS set to a low value, communications are longer but do not impact the communication time that much.
	In that case it is better to post the recv very early, so that it can be fully overlapped.
	This communication scheduler does exactly this, doing a topological sort that posts RECVs as soon as possible

	.. warning::
		This scheduler CAN DEADLOCK. It moves a RECV earlier as long as no path in the data
		dependency graph forbids it, which misses the constraint a computation creates when it
		synchronizes the device: the rank then cannot step over it until the peers have posted the
		matching SEND of every RECV issued since its previous computation. Replaying that worst case
		shows it hanging on ZB-H1, ZB-H2, 1F1B and full-remat schedules. It does simulate faster
		than :py:func:`safe_overlap_comm_scheduler` (~2% on 1 wave, ~4-6% on 2 waves), because it
		posts RECVs earlier than any live placement allows, so that margin is only real if no
		computation of the model ever synchronizes. Prefer ``safe-overlap``.
	"""
	compute_ops = [op for op in topological_sort(schedule) if op.op not in comm_types]
	# all fake recvs
	result = [op for op in schedule if op.op in recv_types and op.peer is None]  # fake p2ps

	# Operations that may depend on each other are in the same (block, micro-batch) group
	groups = chains(schedule)

	for comp in compute_ops:
		result.append(comp)
		send_type = {
			OperationType.FORWARD: OperationType.SEND_FORWARD,
			OperationType.BACKWARD_INPUTS: OperationType.SEND_BACKWARD,
		}.get(comp.op, None)
		sends = find_ops(schedule, groups, comp.block_id, comp.mb_id, send_type)
		for node in sends:
			send = schedule[node]
			result.append(send)
			if (recv := matching_comm(schedule, groups, send)) is not None:
				result.append(schedule[recv])

	# Move recv earlier
	# 2 rules:
	# - don't change the order in which recv are done on one rank (no swapping RF2 with RF3) as it could introduce a deadlock
	# - don't start too early: start the recv `prefetch` computations in advance.
	graph = schedule_to_graph(result)
	op_to_node = node_index(result)  # modified later, keep this original mapping
	for i in range(len(result)):
		op = result[i]
		node = op_to_node[op]

		if op.op not in recv_types:
			continue

		j = i - 1
		k = 0
		while j >= 0:
			prevop = result[j]
			if prevop.rank != op.rank:
				j -= 1
				continue
			if prevop.op == op.op:  # same op, same rank ; don't change order
				break
			prevnode = op_to_node[prevop]
			if graph.has_path(node, prevnode):
				break
			if prevop.op not in comm_types:  # computation (real one, skip fake ones)
				if k >= prefetch:
					break
				k += 1

			j -= 1

		# now insert i at index j
		result.insert(j + 1, result.pop(i))

	assert len(result) == len(schedule), f"{len(result)} / {len(schedule)}"
	return result


def toposort_with_recv_prio_comm_scheduler(
	schedule: List[Operation], profiles: List[StageProfile] | None = None
) -> List[Operation]:
	"""
	toposort, but trying to recv as soon as possible + prioritizing operations according to their distance to the next recv
	"""
	graph = schedule_to_graph(schedule)
	op_to_node = node_index(schedule)
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

	# Static heuristic: is this op a (transitive) ancestor of some recv?
	recv_nodes = {op_to_node[op] for op in schedule if op.op in recv_types}

	def compute_dist_to_recv(graph, recv_nodes):
		"""dist[node] = shortest number of forward edges from node to some recv."""
		dist = [math.inf] * graph.num_nodes
		dq = deque()
		for r in recv_nodes:
			dist[r] = 0
			dq.append(r)
		while dq:
			node = dq.popleft()
			for p in graph.get_successors(node):
				if dist[p] == math.inf:
					dist[p] = dist[node] + 1
					dq.append(p)
		return dist

	dist = compute_dist_to_recv(graph, recv_nodes)

	def priority(node):
		return (dist[node], node)

	heap = [(priority(node), node) for node in range(graph.num_nodes) if is_ready(node)]
	heapq.heapify(heap)

	def process_node(node: int):
		result.append(schedule[node])

		for neighbor in graph.get_predecessors(node):
			out_degree[neighbor] -= 1
			if is_ready(neighbor):
				heapq.heappush(heap, (priority(neighbor), neighbor))

		out_degree[node] = -1  # This node is now processed, so it should not be processed again

	result = []
	while heap:
		_, node = heapq.heappop(heap)

		if out_degree[node] == -1:
			continue

		process_node(node)
		if node in pairings:
			process_node(pairings[node])

	assert len(result) == graph.num_nodes, "cycle in dependency graph"
	return result


def schedule_to_graph(schedule: List[Operation]) -> DirectedGraph:
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
		OperationType.RECOMPUTE_FORWARD_FOR_PARAMS,
		OperationType.RECOMPUTE_BACKWARD_INPUTS,
		# Count this as computation to avoid moving them around
		OperationType.PREFETCH_ACTIVATIONS,
	}

	# Create graph with one node per operation
	graph = DirectedGraph(len(schedule))

	# Operations that may depend on each other are in the same (block, micro-batch) group
	groups = chains(schedule)

	# Second pass: add edges
	for i, op in enumerate(schedule):
		if op.op in comm_types:
			# Same micro-batch, peer block, matching operation
			j = matching_comm(schedule, groups, op)
			if j is not None:
				graph.add_edge(i, j)
				graph.add_edge(j, i)

		if op.op in op_to_direct_dependency:
			# Same rank, same micro-batch, sequential dependency (rf->f->sf, rb->b->sb)
			deps = find_ops(schedule, groups, op.block_id, op.mb_id, op_to_direct_dependency[op.op])
			for j in deps:
				graph.add_edge(i, j)

	# Add dependency of AllReduce to last W
	n_micro_batches = max(op.mb_id for op in schedule if op.mb_id is not None)
	for i, op in enumerate(schedule):
		if op.op is OperationType.ALL_REDUCE_PARAM_GRADS:
			w = next(
				o
				for o in groups[op.block_id, n_micro_batches]
				if schedule[o].op is OperationType.BACKWARD_PARAMS
			)
			graph.add_edge(i, w)

	# Add chain dependencies: sequential computations on same processor
	rank_ops: Dict[int, List[int]] = {}

	# Group computations by rank
	for i, op in enumerate(schedule):
		if op.rank not in rank_ops:
			rank_ops[op.rank] = []
		if op.op in compute_ops:
			rank_ops[op.rank].append(i)

	# Add edges between consecutive computations on the same rank
	for _, ops in rank_ops.items():
		for j in range(len(ops)):
			current_op_idx = ops[j]
			if j < len(ops) - 1:
				next_op_idx = ops[j + 1]
				graph.add_edge(next_op_idx, current_op_idx)

	return graph


def add_comms(schedule: List[Operation], signatures: List[Signature]) -> List[Operation]:
	"""
	Add communications to the schedule based on the signatures.
	"""
	_COMPUTE_TO_COMMS = {
		OperationType.FORWARD: (OperationType.RECV_FORWARD, OperationType.SEND_FORWARD),
		OperationType.BACKWARD_INPUTS: (OperationType.RECV_BACKWARD, OperationType.SEND_BACKWARD),
	}
	schedule = schedule.copy()  # avoid mutating in-place

	for op in schedule:
		if op.op in _COMPUTE_TO_COMMS:
			recv_type, send_type = _COMPUTE_TO_COMMS[op.op]
			signature = signatures[op.block_id]
			# Add send and recvs
			sources = (
				sorted(signature.get_all_sources(), reverse=True)
				if recv_type == OperationType.RECV_FORWARD
				else sorted(signature.get_all_targets())
			)
			for src in sources:
				schedule.append(Operation(op.block_id, op.mb_id, recv_type, op.rank, peer=src))

			destinations = (
				sorted(signature.get_all_targets())
				if send_type == OperationType.SEND_FORWARD
				else sorted(signature.get_all_sources(), reverse=True)
			)
			for dst in destinations:
				schedule.append(Operation(op.block_id, op.mb_id, send_type, op.rank, peer=dst))

	# Only stages that compute parameter gradients have something to all-reduce.
	stages = set((op.block_id, op.rank) for op in schedule if op.op is OperationType.BACKWARD_PARAMS)
	for stage, rank in sorted(stages):
		schedule.append(Operation(stage, None, OperationType.ALL_REDUCE_PARAM_GRADS, rank))

	return schedule


class _ScheduleModel:
	"""
	Everything about a schedule that does not depend on where the RECVs are posted.

	A *slot* ``s`` on rank ``r`` means "after computation ``s - 1`` and before computation ``s``"
	of that rank. Since the order of the computations is fixed and a SEND is always posted right
	after the computation producing it, placing the communications of a schedule comes down to
	choosing one slot per RECV.

	:param schedule: schedule containing both computations and communications
	:type schedule: List[Operation]
	:param profiles: one :class:`~elf.profiling.StageProfile` per stage. Without them every
		computation is given the same duration and every transfer none, which is enough to order
		operations but not to time them.
	:type profiles: List[StageProfile] or None
	"""

	#: the computation a communication is tied to on its own rank
	PRODUCED_BY = {
		OperationType.SEND_FORWARD: OperationType.FORWARD,
		OperationType.SEND_BACKWARD: OperationType.BACKWARD_INPUTS,
	}
	CONSUMED_BY = {
		OperationType.RECV_FORWARD: OperationType.FORWARD,
		OperationType.RECV_BACKWARD: OperationType.BACKWARD_INPUTS,
	}

	#: profiling field every computation is timed with; recomputing costs the same as computing
	_TIMED_BY = {
		OperationType.FORWARD: "t_forward",
		OperationType.RECOMPUTE_FORWARD: "t_forward",
		OperationType.RECOMPUTE_FORWARD_FOR_PARAMS: "t_forward",
		OperationType.BACKWARD_INPUTS: "t_backward_inputs",
		OperationType.RECOMPUTE_BACKWARD_INPUTS: "t_backward_inputs",
		OperationType.BACKWARD_PARAMS: "t_backward_params",
	}

	def __init__(self, schedule: List[Operation], profiles: List[StageProfile] | None = None):
		self.ops = topological_sort(schedule)
		groups = chains(self.ops)

		# Match every communication with the one it rendezvous with on the peer rank.
		self.match: Dict[int, int] = {}
		for i, op in enumerate(self.ops):
			if op.op in comm_types and op.peer is not None:
				peer = matching_comm(self.ops, groups, op)
				if peer is not None:
					self.match[i] = peer

		self.computes: Dict[int, List[int]] = defaultdict(list)
		anchor = {}
		for i, op in enumerate(self.ops):
			anchor[i] = len(self.computes[op.rank])
			if op.op not in comm_types:
				self.computes[op.rank].append(i)

		self.compute_index = {
			(op.rank, op.block_id, op.mb_id, op.op): i
			for i, op in enumerate(self.ops)
			if op.op not in comm_types
		}

		# SENDs are pinned right after the computation producing them, RECVs start out right before
		# the computation consuming them, which is the latest they may ever go. Anything else keeps
		# the slot the topological order gave it.
		for i, op in enumerate(self.ops):
			local = None
			if op.op in self.PRODUCED_BY:
				local = self.local(op, self.PRODUCED_BY[op.op])
				anchor[i] = anchor[i] if local is None else anchor[local] + 1
			elif op.op in self.CONSUMED_BY:
				local = self.local(op, self.CONSUMED_BY[op.op])
				anchor[i] = anchor[i] if local is None else anchor[local]
		self.anchor = anchor

		# Nodes of the graph over computations: `base[rank] + k` is the k-th computation of `rank`.
		self.base, offset = {}, 0
		for rank, computes in self.computes.items():
			self.base[rank] = offset
			offset += len(computes)
		self.num_computes = offset

		self.duration = [self._duration(op, profiles) for op in self.ops]
		self.slowdown = [P2P_SLOWDOWN if op.op in self._TIMED_BY else 0.0 for op in self.ops]

	def spin_cost(
		self, timeline: Dict[int, List[float]], rank: int, start: float, end: float
	) -> float:
		"""
		Time the computations of *rank* lose to a p2p kernel in flight from *start* to *end*.

		:param timeline: per rank, the time it reaches each slot, from :py:func:`_reachability`
		:type timeline: Dict[int, List[float]]
		:param rank: the rank the kernel runs on
		:type rank: int
		:param start: when the kernel is posted
		:type start: float
		:param end: when it completes
		:type end: float

		:return: the overlap of the kernel with each computation, weighted by
			:py:data:`P2P_SLOWDOWN`
		:rtype: float
		"""
		reach, computes = timeline[rank], self.computes[rank]
		lost = 0.0
		k = max(bisect_right(reach, start) - 1, 0)  # compute k ends at reach[k + 1]
		while k < len(computes) and reach[k + 1] - self.duration[computes[k]] < end:
			compute = computes[k]
			overlap = min(end, reach[k + 1]) - max(start, reach[k + 1] - self.duration[compute])
			if overlap > 0:
				lost += overlap * self.slowdown[compute]
			k += 1
		return lost

	def local(self, op: Operation, op_type: OperationType) -> int | None:
		"""Node of the *op_type* computation of the same (rank, block, micro-batch) as *op*."""
		return self.compute_index.get((op.rank, op.block_id, op.mb_id, op_type))

	def node(self, rank: int, index: int) -> int:
		"""Node of the *index*-th computation of *rank*."""
		return self.base[rank] + index

	def _duration(self, op: Operation, profiles: List[StageProfile] | None) -> float:
		if op.op in comm_types:
			if op.peer is None or profiles is None:
				return 0.0
			# a RECV_FORWARD and a SEND_BACKWARD carry the output of the previous stage
			block = op.block_id
			if op.op in {OperationType.RECV_FORWARD, OperationType.SEND_BACKWARD} and block > 0:
				block -= 1
			return profiles[block].t_transfer
		field = self._TIMED_BY.get(op.op)
		if field is None:
			return 0.0  # the loss and the gradient all-reduce are not what the pipeline waits for
		return 1.0 if profiles is None else getattr(profiles[op.block_id], field)


def _place_recvs(
	model: _ScheduleModel, targets: Dict[int, int] | None = None, order: List[int] | None = None
) -> Dict[int, int]:
	"""
	Slot of every operation, RECVs placed as close to *targets* as liveness allows.

	The RECVs are placed one at a time, each into the slot of its live interval closest to the slot
	it aims for, as computed on the placements already made. The graph therefore only ever gains
	edges and stays acyclic by construction. ``None`` aims at the earliest live slot.

	No interval is ever empty when the RECVs are placed in an order of execution -- one that
	follows every rank, and puts each transfer after the computation producing it and before the
	one consuming it -- and each RECV goes either at the lower end of its interval or after a
	computation that comes before its transfer in that order. Schedule order is such an order for
	the earliest slots. Targets read off a timeline need the order in which that timeline posts the
	SENDs, which *order* gives.

	:param model: the schedule to place the communications of
	:type model: _ScheduleModel
	:param targets: slot each RECV would like to be posted at, by node
	:type targets: Dict[int, int] or None
	:param order: the RECVs with a peer, in the order to place them; schedule order by default
	:type order: List[int] or None

	:return: the slot of every operation of the schedule, by node
	:rtype: Dict[int, int]
	"""
	ops, match, computes = model.ops, model.match, model.computes
	slot = dict(model.anchor)

	ranks = list(computes)
	column = {rank: c for c, rank in enumerate(ranks)}
	node_rank = [0] * model.num_computes
	node_slot = [0] * model.num_computes
	for rank, local in computes.items():
		for k in range(len(local)):
			node_rank[model.node(rank, k)] = rank
			node_slot[model.node(rank, k)] = k
	predecessors = [[] for _ in range(model.num_computes)]
	successors = [[] for _ in range(model.num_computes)]
	sources = {rank: [] for rank in ranks}  # per rank, sorted indices of nodes with successors

	# The ancestors of a node are closed downwards along every chain, so they are described by one
	# index per rank: `ancestors[n][column[r]]` is the last computation of `r` that `n` depends on
	# (itself included), or -1. Only the first `valid[r]` nodes of each rank are up to date: adding
	# an edge invalidates its descendants, and a query recomputes the chain up to the node asked.
	ancestors = [None] * model.num_computes
	valid = {rank: 0 for rank in ranks}

	def latest_ancestor(node):
		r, k = node_rank[node], node_slot[node]
		base = model.node(r, 0)
		for j in range(valid[r], k + 1):
			highest = list(ancestors[base + j - 1]) if j else [-1] * len(ranks)
			highest[column[r]] = j
			for other in predecessors[base + j]:
				for c, h in enumerate(latest_ancestor(other)):
					if h > highest[c]:
						highest[c] = h
			ancestors[base + j] = highest
			valid[r] = j + 1
		return ancestors[node]

	def earliest_descendants(source):
		"""
		Per rank, index of the first computation that must be issued after `source`; ranks it does
		not reach are left out. Descendants are closed upwards along every chain, so the walk only
		tracks, per rank, the lowest index reached so far, and follows the edges leaving the chain
		above it.
		"""
		lowest, stack = {}, [source]
		while stack:
			node = stack.pop()
			r, k = node_rank[node], node_slot[node]
			last = lowest.get(r, len(computes[r]))
			if k >= last:
				continue
			lowest[r] = k
			base, indices = model.node(r, 0), sources[r]
			for j in indices[bisect_left(indices, k) : bisect_left(indices, last)]:
				stack.extend(successors[base + j])
		return lowest

	def add_edge(source, target):
		predecessors[target].append(source)
		if not successors[source]:
			insort(sources[node_rank[source]], node_slot[source])
		successors[source].append(target)
		for r, k in earliest_descendants(target).items():
			valid[r] = min(valid[r], k)

	if order is None:
		# fake p2p (no peer): nothing to overlap, leave it alone
		order = [i for i, op in enumerate(ops) if op.op in recv_types and i in match]

	last_placed = {}
	for i in order:
		op = ops[i]
		send = match[i]
		sender, rank = ops[send].rank, op.rank
		posted = slot[send]  # the SEND is posted right after computation `posted - 1` of `sender`

		# Earliest: past every computation of `rank` that posting the SEND already depends on.
		lowest = 0
		if op.op == OperationType.RECV_BACKWARD:
			loss = model.local(op, OperationType.LOSS_BACKWARD)
			if loss is not None:
				lowest = slot[loss] + 1
		if posted >= 1:
			lowest = max(lowest, latest_ancestor(model.node(sender, posted - 1))[column[rank]] + 1)

		# Latest: before the computation consuming it, and before any computation of `rank` that
		# the computation following the SEND already depends on.
		latest = slot[i]
		if posted < len(computes[sender]):
			descendants = earliest_descendants(model.node(sender, posted))
			latest = min(latest, descendants.get(rank, math.inf))

		# NCCL matches messages in posting order, so a RECV also cannot overtake the previous one of
		# its stream. In practice `lowest` already grows along a stream and this never binds.
		lowest = max(lowest, last_placed.get((rank, op.op, op.peer), 0))

		assert lowest <= latest, f"{op} has no live slot on rank {rank}: {lowest} > {latest}"
		slot[i] = lowest if targets is None else min(max(targets[i], lowest), latest)
		last_placed[(rank, op.op, op.peer)] = slot[i]

		if slot[send] >= 1 and slot[i] < len(computes[rank]):
			add_edge(model.node(sender, slot[send] - 1), model.node(rank, slot[i]))
		if slot[i] >= 1 and slot[send] < len(computes[sender]):
			add_edge(model.node(rank, slot[i] - 1), model.node(sender, slot[send]))

	return slot


def _serialize_streams(model: _ScheduleModel, slot: Dict[int, int]) -> Dict[int, int]:
	"""
	Hold every RECV back until the computation consuming the previous RECV of its stream, wherever
	that keeps the placement live.

	All the messages of a communicator run on one CUDA stream, so a RECV posted while the previous
	one of its stream is still open cannot start before that one completes. Posting it early buys
	nothing, and it is harmful: until it starts it sits at the head of the device's work queue,
	and when the device has a single queue (``CUDA_DEVICE_MAX_CONNECTIONS=1``, which overlaps
	sends with computations best) every operation issued after it on the rank waits behind it.
	The computation consuming the previous RECV waits for it anyway, so from there on a RECV costs
	nothing. A placement that posts many RECVs ahead, such as the earliest one, otherwise
	serializes whole ranks behind their communications.

	Moving a RECV later only ever strengthens one edge of the liveness graph of
	:py:func:`_place_recvs`: its rank must have issued the computation before the new slot by the
	time the peer issues the computation following the SEND. On the complete graph of a live
	placement, the move keeps it acyclic -- and the placement live -- exactly when that
	computation cannot be reached from the one following the SEND, which is checked for each
	move. RECVs are moved one at a time, last first, so that a RECV never overtakes the next one
	of its stream.

	:param model: the schedule
	:type model: _ScheduleModel
	:param slot: a live placement, from :py:func:`_place_recvs`
	:type slot: Dict[int, int]

	:return: the new placement, by node
	:rtype: Dict[int, int]
	"""
	ops, match, computes = model.ops, model.match, model.computes
	slot = dict(slot)
	successors = defaultdict(lambda: defaultdict(int))  # multigraph over computation nodes

	def comm_edges(i):
		"""The two edges RECV `i` and its SEND add to the graph, for the current slots."""
		send = match[i]
		sender, rank = ops[send].rank, ops[i].rank
		out = []
		if slot[send] >= 1 and slot[i] < len(computes[rank]):
			out.append((model.node(sender, slot[send] - 1), model.node(rank, slot[i])))
		if slot[i] >= 1 and slot[send] < len(computes[sender]):
			out.append((model.node(rank, slot[i] - 1), model.node(sender, slot[send])))
		return out

	recvs = [i for i, op in enumerate(ops) if op.op in recv_types and i in match]
	for i in recvs:
		for a, b in comm_edges(i):
			successors[a][b] += 1

	node_rank, node_slot = {}, {}
	for rank, local in computes.items():
		for k in range(len(local)):
			node_rank[model.node(rank, k)] = rank
			node_slot[model.node(rank, k)] = k

	def earliest_descendant(source, rank):
		"""First computation of `rank` reachable from `source` (chains included), or infinity."""
		lowest = {}
		stack = [source]
		while stack:
			node = stack.pop()
			r, k = node_rank[node], node_slot[node]
			last = lowest.get(r, len(computes[r])) - 1
			if k > last:
				continue
			lowest[r] = k
			for n in range(node, model.node(r, last) + 1):  # the rest of the chain
				stack.extend(successors[n])
		return lowest.get(rank, math.inf)

	streams = defaultdict(list)
	for i in recvs:
		op = ops[i]
		streams[(op.rank, op.block_id, op.op, op.peer)].append(i)

	next_on_stream, previous_consumer = {}, {}
	for stream in streams.values():
		for previous, i in zip(stream, stream[1:]):
			next_on_stream[previous] = i
			consumer = model.local(ops[previous], model.CONSUMED_BY[ops[previous].op])
			if consumer is not None:
				previous_consumer[i] = model.anchor[consumer]

	for i in reversed(recvs):
		if i not in previous_consumer:
			continue
		target = min(previous_consumer[i], model.anchor[i])
		if i in next_on_stream:
			target = min(target, slot[next_on_stream[i]])
		if target <= slot[i]:
			continue
		send = match[i]
		sender, rank = ops[send].rank, ops[i].rank
		if slot[send] < len(computes[sender]):
			# the new edge (rank, target - 1) -> (sender, slot[send]) must not close a cycle
			target = min(target, earliest_descendant(model.node(sender, slot[send]), rank))
		if target <= slot[i]:
			continue
		for a, b in comm_edges(i):
			successors[a][b] -= 1
			if not successors[a][b]:
				del successors[a][b]
		slot[i] = target
		for a, b in comm_edges(i):
			successors[a][b] += 1

	return slot


def _reachability(model: _ScheduleModel, slot: Dict[int, int]) -> Dict[int, List[float]]:
	"""
	Time at which each rank reaches each of its slots, for a given placement.

	Computations are the only thing a rank spends time on, and every communication has a stream of
	its own, so the timeline is a longest path over the computations alone. A computation waits for
	the one before it on its rank, and, when it consumes a transfer, for that transfer to complete:
	the transfer starts once *both* of its sides have been posted -- the SEND right after the
	computation producing it, the RECV at the slot this placement gives it -- and takes the
	transfer time of the tensor. The graph is acyclic whenever the placement is live, which
	:py:func:`_place_recvs` guarantees, so one topological pass is exact.

	:param model: the schedule
	:type model: _ScheduleModel
	:param slot: the slot of every operation, from :py:func:`_place_recvs`
	:type slot: Dict[int, int]

	:return: per rank, the time it reaches each slot (one entry more than it has computations)
	:rtype: Dict[int, List[float]]
	"""
	incoming = [[] for _ in range(model.num_computes)]
	in_degree = [0] * model.num_computes
	outgoing = [[] for _ in range(model.num_computes)]

	def edge(source, target, delay):
		incoming[target].append((source, delay))
		outgoing[source].append(target)
		in_degree[target] += 1

	for rank, computes in model.computes.items():
		for k in range(1, len(computes)):
			edge(model.node(rank, k - 1), model.node(rank, k), model.duration[computes[k - 1]])

	for i, op in enumerate(model.ops):
		if op.op not in model.CONSUMED_BY or i not in model.match:
			continue
		consumer = model.local(op, model.CONSUMED_BY[op.op])
		if consumer is None:
			continue
		consumer = model.node(op.rank, model.anchor[consumer])
		transfer = model.duration[i]
		send = model.match[i]
		# both sides have to be posted for the transfer to start
		for rank, posted in ((op.rank, slot[i]), (model.ops[send].rank, slot[send])):
			if posted >= 1:
				source = model.node(rank, posted - 1)
				if source != consumer:
					edge(source, consumer, model.duration[model.computes[rank][posted - 1]] + transfer)

	reached = [0.0] * model.num_computes
	queue = deque(node for node in range(model.num_computes) if in_degree[node] == 0)
	visited = 0
	while queue:
		node = queue.popleft()
		visited += 1
		for target in outgoing[node]:
			in_degree[target] -= 1
			if in_degree[target] == 0:
				reached[target] = max(
					(reached[source] + delay for source, delay in incoming[target]), default=0.0
				)
				queue.append(target)
	assert visited == model.num_computes, "the placement is not live: its timeline has a cycle"

	# A communication posted at slot k is issued as soon as the rank is done with computation
	# k - 1, whatever computation k is still waiting for.
	timeline = {}
	for rank, computes in model.computes.items():
		reach = [0.0]
		for k, compute in enumerate(computes):
			reach.append(reached[model.node(rank, k)] + model.duration[compute])
		timeline[rank] = reach
	return timeline


def _just_in_time_targets(
	model: _ScheduleModel,
	slot: Dict[int, int],
	timeline: Dict[int, List[float]],
	spin_aware: bool = False,
) -> Dict[int, int]:
	"""
	Slot at which each RECV is posted closest to the time its SEND is.

	A transfer cannot start before both of its sides are posted, so posting a RECV any earlier than
	its SEND buys nothing and only widens the window over which the open communication slows the
	computations down. Posting it later, on the other hand, delays the transfer and everything
	waiting on it. The slot the rank is executing when the SEND goes out -- the last one reached
	before it -- is therefore what to aim for by default.

	That RECV still waits for its SEND during the rest of the computation it is posted before, and
	a waiting p2p kernel slows down the computations next to it as much as a transferring one.
	With *spin_aware*, a RECV may instead aim for the next slot, the first one reached *after* the
	SEND is posted: the SEND then waits for the RECV, on the sender's side, for the rest of the
	receiver's computation. Each RECV takes whichever of the two waits costs less by
	:py:meth:`_ScheduleModel.spin_cost`. When the consumer is at least two slots away, the transfer
	still completes before it starts; when the consumer is that next slot, the time it now starts
	later (a transfer at most) counts against posting late, so the RECV stays just in time unless
	its SEND goes out right at the end of the receiver's computation. A RECV never aims past its
	consumer.

	:param model: the schedule
	:type model: _ScheduleModel
	:param slot: the placement the timeline was measured on
	:type slot: Dict[int, int]
	:param timeline: per rank, the time it reaches each slot, from :py:func:`_reachability`
	:type timeline: Dict[int, List[float]]
	:param spin_aware: whether a RECV may aim after its SEND when that waits less
	:type spin_aware: bool

	:return: the slot each RECV aims for, by node
	:rtype: Dict[int, int]
	"""
	targets = {}
	for i, op in enumerate(model.ops):
		if op.op not in recv_types or i not in model.match:
			continue
		send = model.match[i]
		sender = model.ops[send].rank
		posted = timeline[sender][slot[send]]
		reach = timeline[op.rank]
		# the last slot of this rank reached no later than that, by binary search on a sorted list
		target = max(bisect_right(reach, posted) - 1, 0)
		targets[i] = target

		consumer = model.local(op, model.CONSUMED_BY[op.op])
		if not spin_aware or consumer is None or target + 1 > model.anchor[consumer]:
			continue
		# both waits end when the transfer starts; the transfer itself costs the same either way
		recv_waits = model.spin_cost(timeline, op.rank, reach[target], posted)
		send_waits = model.spin_cost(timeline, sender, posted, reach[target + 1])
		if target + 1 == model.anchor[consumer]:
			# the transfer now starts when the consumer could, instead of during the computation
			# before it: the consumer starts that much later, a transfer at most
			send_waits += CONSUMER_DELAY_WEIGHT * min(model.duration[i], reach[target + 1] - posted)
		if send_waits < recv_waits:
			targets[i] = target + 1
	return targets


def _posting_order(
	model: _ScheduleModel, slot: Dict[int, int], timeline: Dict[int, List[float]]
) -> List[int]:
	"""
	The RECVs with a peer, in the order their SENDs are posted on *timeline*.

	:param model: the schedule
	:type model: _ScheduleModel
	:param slot: the placement the timeline was measured on
	:type slot: Dict[int, int]
	:param timeline: per rank, the time it reaches each slot, from :py:func:`_reachability`
	:type timeline: Dict[int, List[float]]
	:rtype: List[int]
	"""
	recvs = [i for i, op in enumerate(model.ops) if op.op in recv_types and i in model.match]

	def posted(i):
		send = model.match[i]
		return timeline[model.ops[send].rank][slot[send]], i

	return sorted(recvs, key=posted)


def _issue_order(model: _ScheduleModel, slot: Dict[int, int]) -> List[Operation]:
	"""
	Replay the worst case execution of a placement, both to emit the operations in an order every
	rank agrees on, and as an independent check that the placement cannot deadlock.

	:param model: the schedule
	:type model: _ScheduleModel
	:param slot: the slot of every operation, from :py:func:`_place_recvs`
	:type slot: Dict[int, int]

	:return: the schedule, reordered
	:rtype: List[Operation]
	"""
	ops, match = model.ops, model.match
	buckets = {rank: [[] for _ in range(len(c) + 1)] for rank, c in model.computes.items()}
	for i, op in enumerate(ops):
		if op.op in comm_types:
			buckets[op.rank][slot[i]].append(i)

	sequences = {}
	for rank, computes in model.computes.items():
		flat = []
		for s, compute in enumerate(computes):
			# SENDs go first: they are what unblocks the peers
			flat += sorted(buckets[rank][s], key=lambda i: (ops[i].op in recv_types, i))
			flat.append(compute)
		flat += sorted(buckets[rank][len(computes)], key=lambda i: (ops[i].op in recv_types, i))
		sequences[rank] = flat

	position = {rank: 0 for rank in sequences}
	pending = {rank: [] for rank in sequences}
	issued = [False] * len(ops)
	result = []

	progressed = True
	while progressed:
		progressed = False
		for rank, flat in sequences.items():
			while position[rank] < len(flat):
				i = flat[position[rank]]
				if ops[i].op not in comm_types:
					if any(not issued[match[p]] for p in pending[rank]):
						break  # the peers still have to post the matching communications
					pending[rank].clear()
				elif i in match:
					pending[rank].append(i)

				issued[i] = True
				result.append(ops[i])
				position[rank] += 1
				progressed = True

	assert len(result) == len(ops), f"deadlock: only {len(result)} / {len(ops)} issued"
	return result


def _cost(
	model: _ScheduleModel, slot: Dict[int, int], timeline: Dict[int, List[float]]
) -> Tuple[float, float]:
	"""
	What a placement is worth: how long the iteration takes once the computations are slowed down
	by the communications left open next to them, and how much time that slowdown costs overall.

	Every communication is in flight on each of its two ranks from the moment that side is posted
	until the transfer completes, waiting included. :py:meth:`_ScheduleModel.spin_cost` turns that
	into time lost by the computations of the rank, which is added to the time the rank finishes
	at in the timeline. The first term is the latest of those, the second their sum. A
	just-in-time placement is built not to delay any transfer, so the open time is mostly what
	decides between two of them.

	:param model: the schedule
	:type model: _ScheduleModel
	:param slot: the slot of every operation, from :py:func:`_place_recvs`
	:type slot: Dict[int, int]
	:param timeline: per rank, the time it reaches each slot, from :py:func:`_reachability`
	:type timeline: Dict[int, List[float]]

	:return: the slowed-down makespan and the total time lost to open communications, in that
		order, so that comparing two of them prefers the faster placement and then the quieter one
	:rtype: Tuple[float, float]
	"""
	lost = {rank: 0.0 for rank in timeline}
	for i, op in enumerate(model.ops):
		if op.op not in recv_types or i not in model.match:
			continue
		send = model.match[i]
		sender = model.ops[send].rank
		posted = (timeline[op.rank][slot[i]], timeline[sender][slot[send]])
		done = max(posted) + model.duration[i]
		lost[op.rank] += model.spin_cost(timeline, op.rank, posted[0], done)
		lost[sender] += model.spin_cost(timeline, sender, posted[1], done)

	makespan = max((reach[-1] + lost[rank] for rank, reach in timeline.items()), default=0.0)
	return makespan, sum(lost.values())


def safe_overlap_comm_scheduler(
	schedule: List[Operation], profiles: List[StageProfile] | None = None
) -> List[Operation]:
	"""
	Place every RECV as early as possible in the schedule, without ever creating a deadlock.

	The execution model is the following, for each rank:

	- the rank issues its operations one by one, in program order;
	- issuing a communication is asynchronous (it returns immediately), but the communication
	  cannot *complete* before its matching communication has been issued on the peer rank;
	- issuing a computation may block the host until every operation enqueued so far on this rank
	  has completed (a computation is allowed to synchronize the device at any point).

	Together, those rules mean that a rank can only step over a computation once its peers have
	issued the matching communication of every communication it issued since its previous
	computation. Nothing else can ever make a rank wait, so the whole liveness question can be
	written on a graph whose nodes are only the *computations*. Writing ``(r, k)`` for "rank r has
	issued its k-th computation", a communication placed between ``(r, k - 1)`` and ``(r, k)``,
	whose match sits between ``(r', k' - 1)`` and ``(r', k')``, contributes two edges:

	- ``(r', k' - 1) -> (r, k)``, since ``(r, k)`` waits for the match to be posted;
	- ``(r, k - 1) -> (r', k')``, the same rule seen from the peer.

	A placement deadlocks if and only if that graph has a cycle, and since a slot only ever moves
	one pair of edges, the slots that keep it acyclic form an interval, directly computable: the
	RECV cannot be posted before the last computation of its own rank that the SEND already depends
	on (a backward walk from the computation producing the SEND), nor after the first computation
	of its own rank that already depends on the SEND (a forward walk). Both walks stay linear
	because ancestor and descendant sets are closed along each rank's chain, so tracking one index
	per rank is enough to visit every computation at most once.

	RECVs are placed one at a time, earliest first, each into the earliest slot of its interval as
	computed on the placements already made. The graph therefore only ever gains edges and stays
	acyclic by construction, without needing any starting placement to be live -- pinning every
	SEND right after the computation producing it and every RECV right before the computation
	consuming it, the usual placement, is not. Computations keep the order they have in the input
	schedule, and RECVs of the same type and peer keep their relative order on a rank, since NCCL
	matches messages in posting order.

	The earliest slot is not the best one, only the one that is always available: holding a RECV
	back strengthens the second of its two edges, so a placement that aims later is not guaranteed
	to exist -- on some schedules no uniformly capped placement is live, which is why this cannot
	offer a bound on how far ahead RECVs may be posted, and so on how many receive buffers are
	alive at once. :py:func:`timed_overlap_comm_scheduler` aims later where it can, one RECV at a time, by
	clamping its target to this same interval.

	:param schedule: schedule containing both computations and communications
	:type schedule: List[Operation]
	:param profiles: ignored -- this placement is decided by the structure of the schedule alone
	:type profiles: List[StageProfile] or None
	:return: the schedule, reordered
	:rtype: List[Operation]
	"""
	model = _ScheduleModel(schedule)
	return _issue_order(model, _serialize_streams(model, _place_recvs(model)))


def timed_overlap_comm_scheduler(
	schedule: List[Operation], profiles: List[StageProfile] | None = None, rounds: int = 16
) -> List[Operation]:
	"""
	Post every RECV just in time: as late as the timeline allows without delaying its transfer.

	A transfer starts only once both of its sides have been posted, so a RECV posted before its
	SEND buys nothing. All it does is leave a communication open for longer, and an open
	communication slows down every computation that runs next to it -- measurably so, by a few
	percent per communication in flight. A RECV posted *after* its SEND, on the other hand, delays
	the transfer and everything downstream of it, which is what makes a rendezvous placement such
	as ``simulation`` collapse as the pipeline gets wider. The slot the receiving rank is executing
	when the SEND goes out is the one place that pays neither price, and that is where this
	scheduler aims.

	Finding it needs to know how long the operations take, so it works from the
	:class:`~elf.profiling.StageProfile` of each stage, which the pipeline measures at setup. Since
	where the RECVs go changes the very timeline the choice is read off, the two are iterated to a
	fixed point, starting from the placement of :py:func:`safe_overlap_comm_scheduler`:

	1. time the schedule (:py:func:`_reachability`) -- a longest path over the computations, exact
	   for a live placement;
	2. read off the slot every RECV should aim for (:py:func:`_just_in_time_targets`);
	3. place them there, or as close as liveness allows (:py:func:`_place_recvs`).

	Each round is a placement that provably cannot deadlock, so the iteration is free to stop
	anywhere; it keeps the cheapest round by :py:func:`_cost` rather than the last, which also
	means it never does worse than the ``safe-overlap`` placement it starts from. Two or three
	rounds are usually enough, and the loop stops as soon as a round changes nothing.

	The iteration runs twice: once aiming every RECV just in time, and once letting each RECV aim
	for the first slot after its SEND instead when that keeps the p2p kernels from waiting for
	their peer next to a computation for longer (``spin_aware`` in
	:py:func:`_just_in_time_targets`). The cost counts that waiting, weighted by the calibrated
	slowdown it inflicts (:py:data:`P2P_SLOWDOWN`), and the cheapest round of both runs wins.

	Without *profiles* there is no timeline to read, and guessing one (every computation the same
	length) measures worse than not trying, so the scheduler then falls back to
	:py:func:`safe_overlap_comm_scheduler`.

	:param schedule: schedule containing both computations and communications
	:type schedule: List[Operation]
	:param profiles: one :class:`~elf.profiling.StageProfile` per stage, as measured by
		:py:func:`elf.profiling.profile_pipeline`
	:type profiles: List[StageProfile] or None
	:param rounds: how many times to iterate the placement at most
	:type rounds: int

	:return: the schedule, reordered
	:rtype: List[Operation]
	"""
	model = _ScheduleModel(schedule, profiles)
	slot = _place_recvs(model)
	if profiles is None:
		return _issue_order(model, _serialize_streams(model, slot))

	start = slot
	timeline = _reachability(model, slot)
	best, best_cost = slot, _cost(model, slot, timeline)
	for spin_aware in (False, True):
		slot, timeline = start, _reachability(model, start)
		for _ in range(rounds):
			targets = _just_in_time_targets(model, slot, timeline, spin_aware)
			placed = _place_recvs(model, targets, _posting_order(model, slot, timeline))
			if placed == slot:
				logger.debug(f"timed-overlap comm scheduler stopped after {_} rounds")
				break
			slot = placed
			timeline = _reachability(model, slot)
			cost = _cost(model, slot, timeline)
			if cost < best_cost:
				best, best_cost = slot, cost
	if best is start:
		# no round beat the earliest placement: emit it as safe-overlap does
		return _issue_order(model, _serialize_streams(model, best))
	# No _serialize_streams here: the just-in-time placement does not post RECVs far ahead, and
	# holding them back further delays transfers that interleaved schedules wait on (measured:
	# 1f1b with 2 waves on 16 GPUs, 64 micro-batches, 129k -> 85k tok/s with the hold-back).
	return _issue_order(model, best)
