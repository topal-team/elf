"""
Handmade partition methods
"""

import logging

from elf.partitioners.profile import DONT_CUT_HERE

logger = logging.getLogger(__name__)


def split_graph(graph, times, memories, n):
	"""
	Naively splits a graph into roughly equal blocks in terms of time.
	This algorithm does not take into account the memory used or transferred, nor the number of
	tensors crossing block boundaries (see :func:`split_graph_balanced`).

	:param graph: symbolic trace of the module to partition (see torch.fx)
	:type graph: fx.GraphModule
	:param times: information about the weights, i.e. the profiled execution time, of each node
	:type times: Dict[str, float]
	:param memories: unused ; it is only there for consistency with the other partitioning functions
	:type memories: Dict[str, float]
	:param n: number of partitions to create
	:type n: Optional[int]

	:return: ``n`` lists of nodes corresponding to each part
	:rtype: List[List[fx.Node]]
	"""
	nodes = list(graph.graph.nodes)
	total_time = sum(times.get(node.name, 0) for node in nodes)
	target_time = total_time / n

	parts = []
	current_part = []
	current_time = 0

	for node in nodes:
		node_time = times.get(node.name, 0)
		if current_time > target_time * (len(parts) + 1) and len(parts) < n - 1:
			parts.append(current_part)
			current_part = []
		current_part.append(node)
		current_time += node_time
	parts.append(current_part)

	return parts


def _evaluate_partition_balance(parts, times):
	"""
	Evaluates the balance of a partition.
	"""
	if any(len(part) == 0 for part in parts):
		return float("inf")
	loads = [sum(times.get(node.name, 0) for node in part) for part in parts]
	avg_load = sum(loads) / len(loads)
	balance = max(loads) / avg_load
	return balance


def split_graph_balanced(graph, times, memories, n):
	"""
	Splits a graph into contiguous blocks, searching for the best balance in terms of time.
	This algorithm does not take into account the memory used.
	For each maximum number of cuts (tensors entering a block, from 1 to 10), the graph is split
	greedily from the end, and the imbalance margin allowed when closing a block is tuned to
	minimize ``max(load) / mean(load)``. The partition with the best score is returned, with a
	penalty for each additional cut. The cut limit is a preference: a block is extended past its
	time target until the limit is met, and the first block takes whatever remains.

	:param graph: symbolic trace of the module to partition (see torch.fx)
	:type graph: fx.GraphModule
	:param times: dictionary indicating the weight, i.e. the profiled execution time, for each node
	:type times: Dict[str, float]
	:param memories: unused ; it is only there for consistency with the other partitioning functions
	:type memories: Dict[str, float]
	:param n: number of partitions to create
	:type n: int

	:return: ``n`` lists of nodes corresponding to each part
	:rtype: List[List[fx.Node]]
	"""
	best_parts = None
	best_penalized_score = float("inf")
	cut_penalty = 0.15  # arbitrary
	max_cuts = 10

	for cuts in range(1, max_cuts + 1):
		imbalance_ratio = 0
		parts = _split_graph_balanced_once(graph, times, memories, n, imbalance_ratio, cuts)
		score = _evaluate_partition_balance(parts, times)

		while imbalance_ratio < 1:
			imbalance_ratio += 0.01
			new_parts = _split_graph_balanced_once(graph, times, memories, n, imbalance_ratio, cuts)
			new_score = _evaluate_partition_balance(new_parts, times)
			if new_score > score:
				break

			parts = new_parts
			score = new_score

		penalized_score = score + (cuts - 1) * cut_penalty
		if penalized_score < best_penalized_score:
			logger.debug(f"New best score: {penalized_score} with {cuts} cuts")
			best_penalized_score = penalized_score
			best_parts = parts

	return best_parts


def _split_graph_balanced_once(graph, times, memories, n, imbalance_ratio, max_cuts):
	"""
	Creates one partition of the graph, with a given imbalance ratio and maximum number of cuts.
	"""
	nodes = list(graph.graph.nodes)
	total_time = sum(times.get(node.name, 0) for node in nodes)
	target_time = total_time / n
	imbalance_margin = imbalance_ratio * target_time

	parts = [[] for _ in range(n)]
	needed_inputs = []

	current_part = n - 1
	current_time = 0
	for node in reversed(nodes):
		# Go to the next part if we are over the balance, or close to it, and we have at most max_cuts inputs
		if (
			current_part > 0
			and (current_time >= target_time or target_time - current_time < imbalance_margin)
			and len(needed_inputs) <= max_cuts
			and all(memories[x] != DONT_CUT_HERE for x in needed_inputs)  # force no cut here
		):
			current_part -= 1
			current_time = 0
			needed_inputs = []
		parts[current_part].insert(0, node)
		current_time += times.get(node.name, 0)
		for dep in node.all_input_nodes:
			if dep.name not in needed_inputs and dep not in parts[current_part]:
				needed_inputs.append(dep.name)
		if node.name in needed_inputs:
			needed_inputs.remove(node.name)

	return parts
