"""
API and main utils for graph partition
"""

import torch
import numpy as np

from elf.zb_utils import LayerDWTracer
from .profile import profile_operations
from .custom import split_graph, split_graph_constrained
from .metis import split_graph_metis
from .dagP import split_graph_dagP
from .utils import remove_inplace_leaves, Signature
import logging

logger = logging.getLogger("partition")


def check_partition(graph, parts, inputs, outputs):
	original_inputs = list(node.target for node in graph.nodes if node.op == "placeholder")

	if original_inputs != inputs[0]:
		raise Exception(
			f"Inputs of the first part do not match original inputs: {original_inputs} != {inputs[0]}"
		)

	for i in range(1, len(parts)):
		for input in inputs[i]:
			is_in_prev_part = False
			for j in reversed(range(i)):
				if input in outputs[j]:
					is_in_prev_part = True
					break
			if not is_in_prev_part:
				raise Exception(f"Input {input} of part {i} is not an output of any previous part")


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
		for i in reversed(inputs):  # we are inserting before, so we do in reverse order
			node = subgraph.placeholder(i)
			env[node.name] = node

	for node in nodes:
		env[node.name] = subgraph.node_copy(node, load_arg)

	with subgraph.inserting_after():
		subgraph.output(tuple(env[o] for o in outputs))

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
	inputs = {i: [] for i in range(len(parts))}
	outputs = {i: [] for i in range(len(parts))}

	def add_outputs(i, arg):
		if isinstance(arg, (list, tuple)):
			for item in arg:
				add_outputs(i, item)
		elif hasattr(arg, "name"):
			outputs[i].append(arg.name)
		elif isinstance(arg, dict):
			for value in arg.values():
				add_outputs(i, value)

	i = len(parts)
	for part in reversed(parts):
		i -= 1
		to_remove = []
		for node in part:
			if node.op == "placeholder":
				inputs[i].append(node.target)
				to_remove.append(node)
				continue
			elif node.op == "output":
				for arg in node.args:
					add_outputs(i, arg)
				to_remove.append(node)
				continue
			for dep in node.all_input_nodes:
				if dep not in part and dep.name not in inputs[i]:
					if dep.op == "placeholder":
						inputs[i].append(dep.target)
					else:
						inputs[i].append(dep.name)
					if i != 0:
						prev = i - 1
						while prev >= 0:
							if dep in parts[prev]:
								break
							prev -= 1

						if prev == -1:
							continue

						# Return each tensor only once ; sending it to multiple targets is managed later
						if dep.name not in outputs[prev]:
							outputs[prev].append(dep.name)

		for node in to_remove:
			part.remove(node)

	# Fix empty parts
	for i, part in enumerate(parts):
		if len(part) != 0:
			continue
		if i != 0:
			for output in outputs[i - 1]:
				inputs[i].append(output)
				outputs[i].append(output)
		if i != len(parts) - 1:
			for inp in inputs[i + 1]:
				# Special case: since input is a reserved python name,
				# it is replaced in code by input_1. So the dependency will be on input_1 but is fullfilled by input.
				if "input" not in inp and "input" not in inputs[i]:
					inputs[i].append(inp)
				outputs[i].append(inp)

	return inputs, outputs


def get_sources_targets(inputs, outputs):
	"""
	:return:

	        - ``sources``: a list of lists, containing the indices of the source of each input of each part
	        - ``targets``: a list of lists of lists, containing the indices of each target for each output of each part

	:rtype: List[List[int]], List[List[List[int]]]
	"""
	n = len(inputs)
	assert n == len(outputs)
	sources = [[] for _ in range(n)]
	targets = [[] for _ in range(n)]

	# Only one source
	def find_source(inp, imax):
		for i in range(imax):
			if inp in outputs[i]:
				return i
		return None

	# Multiple targets
	def find_targets(out, imin):
		tgts = []
		for i in range(imin + 1, n):
			if out in inputs[i]:
				tgts.append(i)

		# If no target is found, it means the output is the final output
		if not tgts:
			tgts.append(None)
		return tgts

	for i in range(n):
		for inp in inputs[i]:
			src = find_source(inp, i)
			sources[i].append(src)

			# If we do that, we get the target indices in the order of who needs them
			# We want them in the order of the actual output
			# if the model does `return x,y,z` we want targets to be [tgt of x, tgt of y, tgt of z] even if z is needed before x

			# Actually, the way we constructed the graph is that the output nodes are ordered from first to last used (see get_inputs_outputs), so that could work ? TODO: check

			# targets[src].append(i)

		for out in outputs[i]:
			tgts = find_targets(out, i)
			targets[i].append(tgts)

	return sources, targets


def symbolic_trace_with_layerdw(model: torch.nn.Module) -> torch.fx.GraphModule:
	"""Helper function to trace a model with LayerDW modules treated as leaf nodes"""
	tracer = LayerDWTracer()
	graph = tracer.trace(model)
	return torch.fx.GraphModule(model, graph)


def extract_graph_fx(model):
	return symbolic_trace_with_layerdw(model)


def extract_graph_export(model, sample, use_dynamic_batch_size=False):
	if use_dynamic_batch_size:
		dim = torch.export.Dim("batch")
		dynamic_shapes = ({0: dim},)
		exported = torch.export.export(model, args=(sample,), dynamic_shapes=dynamic_shapes)
		module = exported.module()
		return module
	else:
		exported = torch.export.export(model, args=(sample,))
		return exported.module()


def extract_graph(model, sample, mode="fx"):
	if mode == "export":
		return extract_graph_export(model, sample)
	elif mode == "fx":
		return extract_graph_fx(model)
	else:
		raise ValueError(f"Unknown graph extraction mode: {mode}")


def partition_graph(model, n, sample, partitioner="naive"):
	"""
	Splits a graph into n parts of roughly equal time.

	:param model: torch model
	:type model: nn.Module
	:param n: number of parts to create
	:type n: int
	:param sample: example of inputs to feed to the model (used for profiling)
	:type sample: Tensor
	:param partitioner: Different partitioners are available:

	        - naive: does not take into account memory, no constraint on the number of inputs/outputs
	        - constrained: does not take into account memory, inputs & outputs of each block are limited to 1 tensor
	        - metis: uses METIS to minimize both time and communication memory. No hard constraint on inputs/outputs.
	        - dagP: like METIS, but uses dagP to enforce acyclicity of partition.

	:type partitioner: str

	:raise Exception: if the partition is invalid

	:return:

	        - ``n`` new modules corresponding to the partition
	        - name of input variables for each module. Each one of them takes its inputs as named parameters with these names
	        - name of output variables for each module. Each one of them outputs a dictionary with these names as keys

	:rtype: List[fx.GraphModule], List[List[str]], List[List[str]]
	"""
	model.train()
	try:
		graph = extract_graph(model, sample, "fx")

	except Exception as err_fx:
		logger.debug(str(err_fx))
		logger.info("Graph extraction failed with torch.fx. Trying with torch.export.")

		try:
			graph = extract_graph(model, sample, "export")
		except Exception as err_export:
			logger.debug(str(err_export))
			logger.info("Graph extraction using torch.export failed; cannot partition the model.")
			exit(1)

	logger.info(f"Extracted graph has {len(graph.graph.nodes)} nodes")

	times, memories = profile_operations(graph, sample)

	if partitioner == "naive":
		parts = split_graph(graph, times, memories, n)
	elif partitioner == "constrained":
		parts = split_graph_constrained(graph, times, memories, n)
	elif partitioner == "metis":
		parts = split_graph_metis(graph, times, memories, n)
	elif partitioner == "dagP":
		parts = split_graph_dagP(graph, times, memories, n)
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
	# Make sure the order is consistent
	for i in range(n):
		if i != 0:
			inputs[i].sort()
		if i != n - 1:
			outputs[i].sort()

	check_partition(graph.graph, parts, inputs, outputs)
	sources, targets = get_sources_targets(inputs, outputs)
	signatures = [Signature(inputs[i], outputs[i], sources[i], targets[i]) for i in range(n)]

	blocks = []
	for i, p in enumerate(parts):
		subgraph = create_subgraph(graph, p, inputs[i], outputs[i])
		remove_inplace_leaves(subgraph)
		blocks.append(subgraph)
		logger.debug(f"Part {i} - signature = {signatures[i]}")

	estimated_times = [sum([np.median(times.get(n.name, 0)) for n in part]) for part in parts]
	estimated_mems = [sum([memories.get(o, 0) for o in out]) / (2**20) for out in outputs.values()]
	logger.info(f"Estimated times : {['%.3fs' % t for t in estimated_times]}")
	logger.info(f"Estimated memory transfers : {['%.1fMB' % t for t in estimated_mems]}")
	return blocks, signatures
