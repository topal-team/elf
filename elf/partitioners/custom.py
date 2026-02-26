"""
Handmade partition methods
"""

import logging

logger = logging.getLogger(__name__)


def split_graph(graph, times, memories, n):
	"""
	Naively splits a graph into roughly equal blocks in terms of time.
	This algorithm does not take into account the memory used or transferred.
	Unlike split_graph_constrained, the number of input and output tensors of each block is NOT guaranteed.

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


def split_graph_constrained(graph, times, memories, n):
	"""
	Naively splits a graph into roughly equal blocks in terms of time.
	This algorithm does not take into account the memory used or transferred.
	Unlike split_graph, it is guaranteed that every block has 1 tensor as input and 1 tensor as output.
	This algorithm tries different numbers of cuts (1-5), finds the best partition for each,
	and returns the one with the best balance score (with a small penalty for more cuts).

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
	cut_penalty = 0.1
	max_cuts = 10

	for max_cuts in range(1, max_cuts + 1):
		imbalance_ratio = 0
		parts = split_graph_constrained_util(graph, times, memories, n, imbalance_ratio, max_cuts)
		score = _evaluate_partition_balance(parts, times)

		while imbalance_ratio < 1:
			imbalance_ratio += 0.01
			new_parts = split_graph_constrained_util(graph, times, memories, n, imbalance_ratio, max_cuts)
			new_score = _evaluate_partition_balance(new_parts, times)
			if new_score > score:
				break

			parts = new_parts
			score = new_score

		penalized_score = score + (max_cuts - 1) * cut_penalty
		if penalized_score < best_penalized_score:
			logger.debug(f"New best score: {penalized_score} with {max_cuts} cuts")
			best_penalized_score = penalized_score
			best_parts = parts

	return best_parts


def split_graph_constrained_util(graph, times, memories, n, imbalance_ratio, max_cuts):
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
