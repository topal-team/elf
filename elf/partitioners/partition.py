"""
API and main utils for graph partition
"""

import shutil
from typing import Dict, List, Optional

import torch
import torch.fx as fx

from ..registry import PARTITIONERS, resolve
from .profile import profile_operations
from .tracing import trace
from .utils import remove_inplace_leaves, Signature, export_partition_to_image
import logging

logger = logging.getLogger("partition")


class PartitionResult:
	"""
	Result of partitioning a computation graph.

	Wraps all intermediate data and provides convenient access to
	signatures, estimated costs, blocks, and visualization.

	:param graph_module: the traced graph module
	:type graph_module: fx.GraphModule
	:param parts: reordered partition (lists of fx.Node per part)
	:type parts: List[List[fx.Node]]
	:param inputs: input variable names for each part
	:type inputs: Dict[int, List[str]]
	:param outputs: output variable names for each part
	:type outputs: Dict[int, List[str]]
	:param times: profiled execution time per node name
	:type times: Dict[str, float]
	:param memories: profiled memory size per node name
	:type memories: Dict[str, float]
	"""

	def __init__(
		self,
		graph_module: fx.GraphModule,
		parts: List[List[fx.Node]],
		inputs: Dict[int, List[str]],
		outputs: Dict[int, List[str]],
		times: Dict[str, float],
		memories: Dict[str, float],
	):
		self.graph_module = graph_module
		self.parts = parts
		self.inputs = inputs
		self.outputs = outputs
		self.times = times
		self.memories = memories
		self._signatures: Optional[List[Signature]] = None

	@property
	def n(self) -> int:
		"""Number of parts in the partition."""
		return len(self.parts)

	@property
	def signatures(self) -> List[Signature]:
		"""Dataflow signatures for each part."""
		if self._signatures is None:
			sources, targets = get_sources_targets(self.inputs, self.outputs)
			self._signatures = [
				Signature(self.inputs[i], self.outputs[i], sources[i], targets[i]) for i in range(self.n)
			]
		return self._signatures

	@property
	def estimated_times(self) -> List[float]:
		"""Per-part estimated execution times (seconds)."""
		return [sum(self.times.get(node.name, 0) for node in part).item() for part in self.parts]

	@property
	def estimated_memory_transfers(self) -> List[float]:
		"""Per-part estimated memory transfers (MB)."""
		return [sum(self.memories.get(o, 0) for o in self.outputs[i]) / (2**20) for i in range(self.n)]

	@property
	def balance_score(self) -> float:
		"""
		Balance score of the partition, defined as:

		.. math::
			\\max_{p \\in [0, n-1]} \\frac{T_p}{\\frac{1}{n} \\sum_{p=0}^{n-1} T_p}

		Lower is better. The optimal balance score is 1.
		"""
		return max(self.estimated_times) / (sum(self.estimated_times) / self.n)

	@property
	def transfer_score(self) -> float:
		"""Transfer score of the partition. Lower is better."""
		return sum(
			self.estimated_memory_transfers[:-1]
		)  # don't count last part, as it will not need to send its output to anyone

	def create_blocks(self) -> List[fx.GraphModule]:
		"""Create executable subgraph modules from the partition."""
		return _create_subgraphs(self.graph_module, self.parts, self.inputs, self.outputs)

	def validate(self):
		"""
		Check that the partition is valid.

		:raise Exception: if the partition is invalid
		"""
		check_partition(self.graph_module.graph, self.parts, self.inputs, self.outputs)

	def to_image(self, filename: str, format: str = "png") -> str:
		"""
		Export partition visualization to an image file.

		:param filename: output filename
		:param format: image format (e.g. "png", "svg", "pdf")
		:return: path to the generated image
		"""
		return export_partition_to_image(self.parts, filename, format)

	def __repr__(self):
		times_str = ", ".join(f"{t:.3f}s" for t in self.estimated_times)
		return f"PartitionResult(n={self.n}, estimated_times=[{times_str}])"


def _check_for_partitioner(partitioner):
	"""
	Check if the partitioner is installed.

	:param partitioner: the partitioner to check
	:type partitioner: PartitionerFn
	:return: the partitioner if it is installed, otherwise the naive partitioner
	:rtype: PartitionerFn
	"""
	if partitioner == PARTITIONERS["metis"]:
		if not shutil.which("gpmetis"):
			logger.warning("metis is not installed, falling back to naive")
			return PARTITIONERS["naive"]
	elif partitioner == PARTITIONERS["dagP"]:
		if not shutil.which("rMLGP"):
			logger.warning("dagP is not installed, falling back to metis")
			return _check_for_partitioner(PARTITIONERS["metis"])

	return partitioner


def check_partition(graph, parts, inputs, outputs):
	"""
	Check if a partition is valid.
	"""
	original_inputs = list(node.target for node in graph.nodes if node.op == "placeholder")

	if original_inputs != inputs[0]:
		raise Exception(
			f"Inputs of the first part do not match original inputs: {original_inputs} != {inputs[0]}"
		)

	for i in range(1, len(parts)):
		if len(parts[i]) == 0:
			raise Exception(f"Part {i} is empty")
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


def _compute_part_dependencies(parts):
	"""
	Compute inputs, outputs, and dependencies for each part.

	:param parts: partition of a model
	:type parts: List[List[fx.Node]]
	:return: inputs dict, outputs dict, dependencies dict
	:rtype: Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[int, Set[int]]]
	"""
	n = len(parts)
	inputs = {i: [] for i in range(n)}
	outputs = {i: [] for i in range(n)}
	dependencies = {i: set() for i in range(n)}

	# Build a mapping from node to its part index
	node_to_part = {}
	for i, part in enumerate(parts):
		for node in part:
			node_to_part[node] = i

	def add_outputs(i, arg):
		if isinstance(arg, (list, tuple)):
			for item in arg:
				add_outputs(i, item)
		elif hasattr(arg, "name"):
			outputs[i].append(arg.name)
		elif isinstance(arg, dict):
			for value in arg.values():
				add_outputs(i, value)

	# Process each part to identify inputs and outputs
	for i, part in enumerate(parts):
		for node in part:
			# Handle existing placeholder nodes from original graph
			if node.op == "placeholder":
				inputs[i].append(node.target)
				continue

			# Handle existing output nodes from original graph
			if node.op == "output":
				for arg in node.args:
					add_outputs(i, arg)
				continue

			# Find dependencies that come from outside this part
			for dep in node.all_input_nodes:
				if dep not in part:
					if dep.op == "placeholder":
						if dep.target not in inputs[i]:
							inputs[i].append(dep.target)
					else:
						if dep.name not in inputs[i]:
							inputs[i].append(dep.name)

						if dep in node_to_part:
							source_part = node_to_part[dep]
							if dep.name not in outputs[source_part]:
								outputs[source_part].append(dep.name)

							if source_part != i:
								dependencies[i].add(source_part)

	return inputs, outputs, dependencies


def _topological_sort_indices(dependencies):
	"""
	Topologically sort parts based on their dependencies using Kahn's algorithm.

	:param dependencies: dict mapping part index to set of part indices it depends on
	:type dependencies: Dict[int, Set[int]]
	:return: list of part indices in topological order
	:rtype: List[int]
	"""
	n = len(dependencies)
	in_degree = {i: len(dependencies[i]) for i in range(n)}
	queue = [i for i in range(n) if in_degree[i] == 0]
	sorted_indices = []

	while queue:
		queue.sort()
		current = queue.pop(0)
		sorted_indices.append(current)

		for i in range(n):
			if current in dependencies[i]:
				in_degree[i] -= 1
				if in_degree[i] == 0:
					queue.append(i)

	if len(sorted_indices) != n:
		raise Exception("Cycle detected in partition dependencies")

	return sorted_indices


def _reorder_by_indices(items, indices):
	"""
	Reorder a list or dict by a list of indices.

	:param items: list or dict to reorder
	:param indices: list of indices in desired order
	:return: reordered list or dict
	"""
	if isinstance(items, list):
		return [items[idx] for idx in indices]
	elif isinstance(items, dict):
		return {new_idx: items[old_idx] for new_idx, old_idx in enumerate(indices)}
	else:
		raise TypeError(f"Unsupported type for reordering: {type(items)}")


def _remove_placeholder_output_nodes(parts):
	"""
	Remove placeholder and output nodes from parts.

	:param parts: partition of a model
	:type parts: List[List[fx.Node]]
	:return: new parts without placeholder/output nodes
	:rtype: List[List[fx.Node]]
	"""
	cleaned_parts = []
	for part in parts:
		cleaned_part = [node for node in part if node.op not in ("placeholder", "output")]
		cleaned_parts.append(cleaned_part)
	return cleaned_parts


def get_inputs_outputs(parts):
	"""
	Compute inputs and outputs for each part, reorder parts topologically,
	and remove placeholder/output nodes.

	:param parts: partition of a model
	:type parts: List[List[fx.Node]]
	:return: reordered parts, inputs dict, outputs dict
	:rtype: Tuple[List[List[fx.Node]], Dict[int, List[str]], Dict[int, List[str]]]
	"""
	# Compute dependencies between parts
	inputs, outputs, dependencies = _compute_part_dependencies(parts)

	# Topologically sort parts
	sorted_indices = _topological_sort_indices(dependencies)

	# Reorder everything according to topological order
	parts = _reorder_by_indices(parts, sorted_indices)
	inputs = _reorder_by_indices(inputs, sorted_indices)
	outputs = _reorder_by_indices(outputs, sorted_indices)

	# Remove placeholder and output nodes
	parts = _remove_placeholder_output_nodes(parts)

	# Sort inputs/outputs for consistency
	for i in range(len(parts)):
		inputs[i].sort()
		outputs[i].sort()

	return parts, inputs, outputs


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


def _dispatch_partitioner(graph, times, memories, n, partitioner):
	"""
	Resolve and run the selected partitioner.
	"""
	partitioner = resolve(partitioner, PARTITIONERS)
	partitioner = _check_for_partitioner(partitioner)
	return partitioner(graph, times, memories, n)


def _create_subgraphs(graph_module, parts, inputs, outputs):
	"""
	Create subgraph modules from partitioned nodes.

	:param graph_module: original graph module
	:param parts: partitioned graph nodes
	:param inputs: inputs dict for each part
	:param outputs: outputs dict for each part
	:return: list of subgraph modules
	:rtype: List[fx.GraphModule]
	"""
	blocks = []
	for i, part in enumerate(parts):
		subgraph = create_subgraph(graph_module, part, inputs[i], outputs[i])
		remove_inplace_leaves(subgraph)
		blocks.append(subgraph)
	return blocks


def split(graph_module, times, memories, n, partitioner="constrained") -> PartitionResult:
	"""
	Partition a traced graph into ``n`` parts and return a :class:`PartitionResult`.

	:param graph_module: traced graph module (from :func:`trace`)
	:type graph_module: fx.GraphModule
	:param times: profiled execution time per node name (from :func:`profile_operations`)
	:type times: Dict[str, float]
	:param memories: profiled memory size per node name (from :func:`profile_operations`)
	:type memories: Dict[str, float]
	:param n: number of parts
	:type n: int
	:param partitioner: partitioner to use (name or callable)
	:type partitioner: str or Partitioner
	:return: partition result
	:rtype: PartitionResult
	"""
	raw_parts = _dispatch_partitioner(graph_module, times, memories, n, partitioner)
	assert len(raw_parts) == n, f"Expected {n} parts, got {len(raw_parts)}"

	parts, inputs, outputs = get_inputs_outputs(raw_parts)
	result = PartitionResult(graph_module, parts, inputs, outputs, times, memories)

	result.validate()

	return result


def partition(model, n, sample, partitioner="constrained", tracer="default") -> PartitionResult:
	"""
	Trace, profile, and partition a model in one call.

	:param model: torch model
	:type model: nn.Module
	:param n: number of parts to create
	:type n: int
	:param sample: example input (used for tracing and profiling)
	:type sample: Tensor
	:param partitioner: partitioner to use (name or callable)
	:type partitioner: str or Partitioner
	:param tracer: tracer to use (name or callable)
	:type tracer: str or Tracer
	:return: partition result
	:rtype: PartitionResult
	"""
	model.train()

	graph_module = trace(model, sample, tracer)
	logger.info(f"Extracted graph has {len(graph_module.graph.nodes)} nodes")

	times, memories = profile_operations(graph_module, sample)

	result = split(graph_module, times, memories, n, partitioner)

	logger.info(f"Estimated times: {['%.3fs' % t for t in result.estimated_times]}")
	logger.info(
		f"Estimated memory transfers: {['%.1fMB' % m for m in result.estimated_memory_transfers]}"
	)

	for i in range(result.n):
		logger.debug(f"Part {i} - signature = {result.signatures[i]}")

	return result
