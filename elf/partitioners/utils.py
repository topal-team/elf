"""
Utility functions for partitioning
"""

import sys

import torch

__all__ = [
	"remove_inplace_leaves",
	"signatures_from_sources_targets",
	"get_sources_targets_sequential",
	"Signature",
]


def remove_dupes(seq):
	"""
	Remove duplicates from a list while preserving order
	"""
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]


class Signature:
	def __init__(self, inputs, outputs, sources, targets):
		"""
		A signature is a description of the forward function of a block in a partition, in terms of its inputs and outputs.
		It also contains information about the source block (where does it come from) for each input, and the target blocks (who needs it) for each output.

		:param inputs: input variables
		:type inputs: List[str]
		:param outputs: output variables
		:type outputs: List[str]
		:param sources: source block id for each input variable
		:type sources: List[int]
		:param targets: list of target block ids for each output variable
		:type targets: List[List[int]]
		"""
		self.inputs = inputs
		self.ninputs = len(inputs)
		self.outputs = outputs
		self.noutputs = len(outputs)

		self.sources = sources
		self.targets = targets

		assert len(sources) == self.ninputs
		assert len(targets) == self.noutputs

	def get_all_sources(self):
		"""
		Set of all blocks that feed into this one
		"""
		return remove_dupes(self.sources)

	def get_all_targets(self):
		"""
		Set of all blocks that need this one's output
		"""
		return remove_dupes(t for tgt in self.targets for t in tgt)

	def __str__(self):
		s = "Signature("
		s += ", ".join([f"{i} ({t})" for i, t in zip(self.inputs, self.sources)])
		s += ") -> ("
		s += ", ".join([f"{o} ({t})" for o, t in zip(self.outputs, self.targets)])
		s += ")"
		return s

	def __repr__(self):
		return str(self)


def signatures_from_sources_targets(sources, targets):
	"""
	Create signatures from sources and targets
	"""
	assert len(sources) == len(targets), "Sources and targets must have the same length"

	signatures = []
	for i in range(len(sources)):
		inputs = sorted(list(sources[i].keys()))
		outputs = sorted(list(targets[i].keys()))
		signatures.append(
			Signature(inputs, outputs, [sources[i][j] for j in inputs], [targets[i][j] for j in outputs])
		)

	return signatures


def get_sources_targets_sequential(placement):
	"""
	Generates sources and targets for a fully sequential model (no skip connections), with one input and one output per stage.
	This is intended to be used with the ``partitioner=False`` option.

	.. note::
		here's an example of what the returned sources and targets look like:

		.. code-block:: python

			>>> sources = {
				0: { # stage 0's sources
					"input": None # variable 'input' comes from None
				},
				1: {
					"x": 0 # variable 'x' comes from stage 0
				}, ...
			}
			>>> targets = {
				0: { # stage 0's targets
					"output": [1, 2] # variable 'output' goes to stages 1 and 2
				},
				1: {
					"output": [2] # variable 'output' goes to stage 2
				}, ...
			}

	:param placement: placement of the model blocks on gpus
	:type placement: List[int]
	:return: Sources and targets for each stage
	:rtype: Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, List[int]]]]
	"""
	sources = {}
	targets = {}
	for i in range(len(placement)):
		# Everyone needs full signatures to generate schedule
		sources[i] = {"input": i - 1 if i != 0 else None}
		targets[i] = {"output": [i + 1 if i != len(placement) - 1 else None]}
	return sources, targets


def remove_inplace_leaves(module):
	"""
	Remove in-place operations from leaf nodes and their views in the module's graph.

	This is necessary because in-place operations on leaves or views of leaves
	can break gradient computation.

	:param module: The module to modify
	:type module: torch.fx.GraphModule
	"""
	# First pass: identify leaf nodes (nodes with no non-placeholder dependencies)
	leaves = set()
	for node in module.graph.nodes:
		if node.op in ["placeholder", "output"]:
			continue
		deps = [d for d in node.all_input_nodes if d.op != "placeholder"]
		if len(deps) == 0:
			leaves.add(node)

	# Second pass: track views of leaves (including views of views, etc.)
	# A node is considered a view of a leaf if it's created by a view operation
	# (operation that returns an alias) on a leaf or another view of a leaf
	# Use fixed-point iteration to handle arbitrary depth of view chains
	views_of_leaves = set()
	changed = True
	while changed:
		changed = False
		for node in module.graph.nodes:
			if node in views_of_leaves:
				continue  # Already identified
			if node.op == "call_function" and _returns_alias(node.target):
				# Check if any input is a leaf or view of a leaf
				for arg in node.all_input_nodes:
					if arg in leaves or arg in views_of_leaves:
						views_of_leaves.add(node)
						changed = True
						break

	# Third pass: remove in-place operations on leaves and views of leaves
	nodes_to_fix = leaves | views_of_leaves
	for node in module.graph.nodes:
		if node in nodes_to_fix:
			if node.op == "call_function":
				if _is_inplace_operation(node.target):
					new_target = _get_outplace_equivalent(node.target)
					node.target = new_target
			elif node.op == "call_module":
				submodule = module.graph.owning_module.get_submodule(node.target)
				if submodule is not None and getattr(submodule, "inplace", False):
					setattr(submodule, "inplace", False)

	module.recompile()


def _returns_alias(target):
	"""
	Check if an operation returns an alias (view) of its input tensor.

	Uses PyTorch's schema system to detect operations that return tensors
	sharing storage with their inputs.

	:param target: The target function or operation
	:return: True if the operation returns an alias/view
	:rtype: bool
	"""
	# For OpOverload (from torch.export), check the schema for alias annotations
	if hasattr(target, "_schema"):
		try:
			schema_str = str(target._schema)
			# PyTorch's schema uses annotations like:
			# -> Tensor(a) : returns an alias of input annotated with 'a'
			# -> Tensor(a!): returns a mutated alias
			# Check if return type contains any alias annotation (a, b, c, etc.)
			# Pattern: "-> Tensor(X" or "-> (Tensor(X" where X is a letter
			import re

			# Match patterns like "-> Tensor(a)" or "-> (Tensor(a"
			# The alias annotation is a single letter in parentheses after Tensor
			pattern = r"-> (?:\()?Tensor\([a-z]"
			if re.search(pattern, schema_str):
				return True
		except (AttributeError, TypeError):
			pass

	return False


def _is_inplace_operation(target):
	"""
	Check if a function target represents an in-place operation.

	Uses PyTorch's schema system when available (`is_mutable`), otherwise
	falls back to the naming convention (trailing underscore).

	:param target: The target function or operation
	:return: True if the operation is in-place
	:rtype: bool
	"""
	# Primary method: use schema's is_mutable (works with torch.export)
	if hasattr(target, "_schema"):
		try:
			return target._schema.is_mutable
		except AttributeError:
			pass

	# Fallback: PyTorch convention - in-place operations end with underscore
	# This works for both symbolic_trace and ATen ops
	if hasattr(target, "__name__"):
		name = target.__name__
		# For ATen ops, extract base name (e.g., "relu_" from "relu_.default")
		if "." in name:
			name = name.split(".")[0]
		return name.endswith("_")

	return False


def _get_outplace_equivalent(target):
	"""
	Get the out-of-place equivalent of an in-place operation.

	Converts operations like relu_ to relu, add_ to add, etc.

	:param target: The in-place operation target
	:return: The out-of-place equivalent operation
	"""
	if not hasattr(target, "__name__"):
		raise ValueError(f"Cannot find out-of-place equivalent for {target}: no __name__ attribute")

	op_name = target.__name__
	# Extract base name (e.g., "relu_" from "relu_.default" or just "relu_")
	base_name = op_name.split(".")[0]

	if not base_name.endswith("_"):
		raise ValueError(f"Operation {op_name} does not appear to be in-place (no trailing underscore)")

	# Remove trailing underscore to get out-of-place name
	outplace_name = base_name[:-1]

	# Handle ATen ops (torch.ops.aten.operation_.overload)
	if hasattr(target, "_overloadname"):
		try:
			outplace_op = getattr(torch.ops.aten, outplace_name)
			return getattr(outplace_op, target._overloadname)
		except AttributeError:
			raise ValueError(
				f"Could not find ATen operation torch.ops.aten.{outplace_name}.{target._overloadname}"
			)

	# Handle regular torch functions (torch.relu_)
	if hasattr(torch, outplace_name):
		return getattr(torch, outplace_name)

	raise ValueError(f"Could not find out-of-place equivalent for {target}")


def prune_graph(graph):
	"""
	Remove all dead-ends in the graph, i.e. all nodes that are not in the path from any input to any output.

	.. warning::
		This function is correct only if the graph has been functionalized, i.e. no nodes have side effects.
		Pruning the graph also deletes guards as they are not part of the computation, which can be unsafe.

	:param graph: The graph to prune
	:type graph: torch.fx.GraphModule
	:return: The pruned graph
	:rtype: torch.fx.GraphModule
	"""
	original_recursion_limit = sys.getrecursionlimit()
	sys.setrecursionlimit(
		max(len(graph.graph.nodes) * 2, original_recursion_limit)
	)  # avoid random stack overflows
	reachable = set()

	def mark_reachable(node):
		"""Recursively mark a node and all its inputs as reachable"""
		if node in reachable:
			return
		reachable.add(node)
		for input_node in node.all_input_nodes:
			mark_reachable(input_node)

	# Start from output nodes and mark all nodes that contribute to outputs
	for node in graph.graph.nodes:
		if node.op == "output":
			mark_reachable(node)

	# Remove unreachable nodes in reverse order to ensure users are erased before their dependencies
	for node in reversed(list(graph.graph.nodes)):
		if node not in reachable and node.op not in ["placeholder", "output"]:
			graph.graph.erase_node(node)

	graph.recompile()

	sys.setrecursionlimit(original_recursion_limit)
	return graph
