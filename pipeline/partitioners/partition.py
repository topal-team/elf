"""
API and main utils for graph partition
"""

import torch
import numpy as np
from .profile import profile_operations
from .custom import split_graph, split_graph_constrained
from .metis import split_graph_metis
from .dagP import split_graph_dagP

import logging

logger = logging.getLogger("partition")


def create_subgraph(graph_module, nodes, inputs, outputs):
	"""
	Creates a module from one block of a partition.
	The module takes as input a dictionary {inputs1: value1, inputs2: value2, ..} where the keys are the values in parameter inputs.
	and returns a similar dictionary, with keys from parameter outputs.

	:return: a graph that can be used like a nn.Module
	:rtype: fx.GraphModule
	"""
	subgraph = torch.fx.Graph()
	env = {}

	def load_arg(a):
		return env[a.name]

	with subgraph.inserting_before():
		for i in inputs:
			node = subgraph.placeholder(i)
			env[node.name] = node

	for node in nodes:
		env[node.name] = subgraph.node_copy(node, load_arg)

	with subgraph.inserting_after():
		subgraph.output({o: env[o] for o in outputs})

	return torch.fx.GraphModule(graph_module, subgraph)


def get_inputs_outputs(parts):
	"""
	Finds the dependencies between each block of a partition.
	Removes inputs/outputs nodes of each part to merge them into 2 nodes.

	:param parts: partition of a model
	:type parts: List[List[fx.Node]]
	:return: 2 dicts, one for inputs and one for outputs, respectively. Each one has the format: {partition_idx: [target1, target2, ..]} where targets are node names.
	:rtype: Dict[int, List[str]], Dict[int, List[str]]
	"""
	inputs = {i: set() for i in range(len(parts))}
	outputs = {i: set() for i in range(len(parts))}

	def add_outputs(i, arg):
		if isinstance(arg, (list, tuple)):
			for item in arg:
				add_outputs(i, item)
		elif hasattr(arg, "name"):
			outputs[i].add(arg.name)
		elif isinstance(arg, dict):
			for value in arg.values():
				add_outputs(i, value)

	i = len(parts)
	for part in reversed(parts):
		i -= 1
		for node in part:
			if node.op == "placeholder":
				inputs[i].add(node.target)
				part.remove(node)
				continue
			elif node.op == "output":
				for arg in node.args:
					add_outputs(i, arg)
				part.remove(node)
				continue
			for dep in node.all_input_nodes:
				if dep not in part and dep.name not in inputs[i]:
					inputs[i].add(dep.name)
					if dep not in parts[i - 1] and dep.name not in outputs[i - 1]:
						raise Exception(
							f"Skip connection detected in partition. Node {node} is located in part {i} and needs output of node {dep} which is not in part {i - 1}."
						)
					outputs[i - 1].add(dep.name)

	# Fix empty parts
	for i, part in enumerate(parts):
		if len(part) != 0:
			continue
		if i != 0:
			for output in outputs[i - 1]:
				inputs[i].add(output)
				outputs[i].add(output)
		if i != len(parts) - 1:
			for inp in inputs[i + 1]:
				inputs[i].add(inp)
				outputs[i].add(inp)

	return inputs, outputs


def get_inputs_outputs_single(part):
	"""
	Get the input/output nodes of a single part from the graph
	Removes inputs/outputs nodes of each part. They should be added back later.
	Useful for already splitted graphs

	:param part: one part of a partitioned model
	:type part: List[fx.Node]

	:return: names of the input and output variables
	:rtype: List[str], List[str]
	"""
	inputs = set()
	outputs = set()

	def add_outputs(arg):
		if isinstance(arg, (list, tuple)):
			for item in arg:
				add_outputs(item)
		elif hasattr(arg, "name"):
			outputs.add(arg.name)
		elif isinstance(arg, dict):
			for value in arg.values():
				add_outputs(value)

	for node in part:
		if node.op == "placeholder":
			inputs.add(node.target)
			part.remove(node)
		if node.op == "output":
			for arg in node.args:
				add_outputs(arg)
			part.remove(node)
	return part, inputs, outputs


def duplicate_symsizes(graph, times, memories):
	"""
	Duplicates symbolic size nodes in the graph to avoid sharing between different parts of the model. Having a unique ``sym_size`` node creates dependencies from the entire graph, making it impossible to partition it without long-range connections.

	:param graph: The computational graph to modify
	:type graph: torch.fx.Graph
	:param times: A dictionary mapping node names to their execution times
	:type times: Dict[str, float]
	:param memories: A dictionary mapping node names to their memory usage
	:type memories: Dict[str, int]

	:return: None. The function modifies the graph in-place.
	"""
	i = 0
	for node in graph.nodes:
		if node.name == "sym_size_int":
			to_replace = {}
			for user in node.users:
				i += 1
				with graph.inserting_before(user):
					new_sym_size = node.graph.create_node(
						"call_function", torch.ops.aten.sym_size, (user.args[0], node.args[1]), {}
					)
				to_replace[user] = new_sym_size

			for user, new_sym_size in to_replace.items():
				user.replace_input_with(node, new_sym_size)
				times[new_sym_size.name] = times[node.name]
				memories[new_sym_size.name] = memories[node.name]

			graph.erase_node(node)
			i -= 1
	logger.info(f"Symsize duplication created {i} nodes.")


def partition_graph(model, n, sample, mode="naive"):
	"""
	Splits a graph into n parts of roughly equal time.

	:param model: torch model
	:type model: nn.Module
	:param n: number of parts to create
	:type n: int
	:param sample: example of inputs to feed to the model (used for profiling)
	:type sample: Tensor
	:param mode: Different modes are available:

	        - naive: does not take into account memory, no constraint on the number of inputs/outputs
	        - constrained: does not take into account memory, inputs & outputs of each block are limited to 1 tensor
	        - metis: uses METIS to minimize both time and communication memory. No hard constraint on inputs/outputs.
	        - dagP: like METIS, but uses dagP to enforce acyclicity of partition.

	:type mode: str

	:return:

	        - ``n`` new modules corresponding to the partition
	        - name of input variables for each module. Each one of them takes its inputs as named parameters with these names
	        - name of output variables for each module. Each one of them outputs a dictionary with these names as keys

	:rtype: List[fx.GraphModule], List[List[str]], List[List[str]]
	"""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	try:
		trace = torch.fx.symbolic_trace(model)

	except Exception as err_fx:
		logger.info("Graph extraction failed with torch.fx. Cannot partition the model.")
		logger.debug(str(err_fx))

		try:
			dim = torch.export.Dim("batch")
			dynamic_shapes = ({0: dim},)
			trace = torch.export.export(model, args=(sample,), dynamic_shapes=dynamic_shapes).module()
		except Exception as err_export:
			logger.info("Graph extraction using torch.export failed; falling back to torch.fx.")
			logger.debug(str(err_export))
			exit(1)

	logger.info(f"Traced module has {len(trace.graph.nodes)} nodes")

	sample = sample.to(device)
	times, memories = profile_operations(trace, sample)

	duplicate_symsizes(trace.graph, times, memories)
	trace.recompile()

	if mode == "naive":
		parts = split_graph(trace, times, memories, n)
	elif mode == "constrained":
		parts = split_graph_constrained(trace, times, memories, n)
	elif mode == "metis":
		parts = split_graph_metis(trace, times, memories, n)
	elif mode == "dagP":
		parts = split_graph_dagP(trace, times, memories, n)
	else:
		raise Exception(
			"Unknown graph partitioning mode : {mode}.\n\
						Available modes:\n\t\
						- naive: does not take into account memory, no constraint on the number of inputs/outputs\n\t\
						- constrained: does not take into account memory, inputs & outputs of each block are limited to 1 tensor\n\t\
						- metis: uses METIS to minimize both time and communication memory. No hard constraint on inputs/outputs.\n\t\
						- dagP: like METIS, but uses dagP to enforce acyclicity of partition."
		)

	while len(parts) != n:
		parts.append([])

	inputs, outputs = get_inputs_outputs(parts)

	blocks = []
	for i, p in enumerate(parts):
		graph = create_subgraph(trace, p, inputs[i], outputs[i])
		blocks.append(graph)

	estimated_times = [sum([np.median(times.get(n.name, 0)) for n in part]) for part in parts]
	estimated_mems = [sum([memories.get(o, 0) for o in out]) / (2**20) for out in outputs.values()]
	logger.info(f'Estimated times : {["%.3fs" % t for t in estimated_times]}')
	logger.info(f'Estimated memory transfers : {["%.1fMB" % t for t in estimated_mems]}')
	return blocks, inputs, outputs
