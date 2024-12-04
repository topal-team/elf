"""
Handmade partition methods
"""

import numpy as np


def split_graph(graph, times, memories, n=3):
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
	total_time = sum(np.median(times.get(node.name, 0)) for node in nodes)
	target_time = total_time / n

	parts = []
	current_part = []
	current_time = 0

	for node in nodes:
		node_time = np.median(times.get(node.name, 0))
		if current_time + node_time > target_time * (len(parts) + 1) and len(parts) < n:
			parts.append(current_part)
			current_part = []
		current_part.append(node)
		current_time += node_time
	parts.append(current_part)

	return parts


def split_graph_constrained(graph, times, memories, n=3):
	"""
	Naively splits a graph into roughly equal blocks in terms of time.
	This algorithm does not take into account the memory used or transferred.
	Unlike split_graph, it is guaranteed that every block has 1 tensor as input and 1 tensor as output.

	:param graph: symbolic trace of the module to partition (see torch.fx)
	:type graph: fx.GraphModule
	:param times: dictionary indicating the weight, i.e. the profiled execution time, for each node
	:type times: Dict[str, float]
	:param memories: unused ; it is only there for consistency with the other partitioning functions
	:type memories: Dict[str, float]
	:param n: number of partitions to create
	:type n: Optional[int]

	:return: ``n`` lists of nodes corresponding to each part
	:rtype: List[List[fx.Node]]
	"""
	nodes = list(graph.graph.nodes)
	total_time = sum(np.median(times.get(node.name, 0)) for node in nodes)
	target_time = total_time / n

	parts = [[] for _ in range(n)]
	needed_inputs = []

	current_part = n - 1
	current_time = 0
	for node in reversed(nodes):
		if current_time > target_time and len(needed_inputs) <= 1:
			current_part -= 1
			current_time = 0
			needed_inputs = []
		parts[current_part].insert(0, node)
		current_time += np.median(times.get(node.name, 0))
		for dep in node.all_input_nodes:
			if dep.name not in needed_inputs and dep not in parts[current_part]:
				needed_inputs.append(dep.name)
		if node.name in needed_inputs:
			needed_inputs.remove(node.name)

	# Fill the pipeline with identity functions to match number of parts
	# while current_part >= 0:

	return parts
