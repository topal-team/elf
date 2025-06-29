"""
Utility functions for partitioning
"""

import torch

__all__ = ["remove_inplace_leaves"]


def remove_inplace_leaves(module):
	"""
	Remove in-place operations from leaf nodes in the module's graph.

	:param module: The module to modify
	:type module: torch.fx.GraphModule
	"""
	for node in module.graph.nodes:
		if node.op in ["placeholder", "output"]:
			continue
		deps = [d for d in node.all_input_nodes if d.op != "placeholder"]
		if len(deps) == 0:
			# Hacky: we assume that all inplace functions end with "_"
			# and have an out-of-place equivalent without the "_"
			if node.op == "call_function" and node.target.__name__[-1] == "_":
				out_of_place_func = getattr(torch, node.target.__name__[:-1])
				node.target = out_of_place_func
			elif node.op == "call_module":
				submodule = module.graph.owning_module.get_submodule(node.target)
				if submodule is not None and getattr(submodule, "inplace", False):
					setattr(submodule, "inplace", False)

	module.recompile()


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
		sources = {
			0: { # stage 0's sources
				"input": None # variable 'input' comes from None
			},
			1: {
				"x": 0 # variable 'x' comes from stage 0
			}, ...
		}
		targets = {
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
