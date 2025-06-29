"""
Extraction of computation graphs from models.
"""

import torch
import torch.nn as nn

import logging

from elf.zb_utils import LayerDWTracer

logger = logging.getLogger(__name__)


def symbolic_trace_with_layerdw(model: torch.nn.Module) -> torch.fx.GraphModule:
	"""Helper function to trace a model with LayerDW modules treated as leaf nodes"""
	tracer = LayerDWTracer()
	graph = tracer.trace(model)
	return torch.fx.GraphModule(model, graph)


def extract_graph_fx(model):
	return symbolic_trace_with_layerdw(model)


def extract_graph_export(model, sample, use_dynamic_batch_size=False):
	"""
	Extract graph using torch.export.

	:param model: Model to extract graph from
	:type model: nn.Module
	:param sample: Sample input to use for tracing
	:type sample: torch.Tensor
	:param use_dynamic_batch_size: Whether to use dynamic batch size
	:type use_dynamic_batch_size: bool
	:return: Exported module
	:rtype: torch.export.ExportedProgram
	"""
	if use_dynamic_batch_size:
		dim = torch.export.Dim("batch")
		dynamic_shapes = ({0: dim},)
		exported = torch.export.export(model, args=(sample,), dynamic_shapes=dynamic_shapes)
		module = exported.module()
		return module
	else:
		exported = torch.export.export(model, args=(sample,))
		return exported.module()


def extract_graph_fx_safe(model, sample):
	"""
	Extract graph using torch.fx, by marking every non-traceable submodule as leaf, iteratively.

	:param model: Model to extract graph from
	:type model: nn.Module
	:param sample: Sample input to use for tracing
	:type sample: torch.Tensor
	:return: Graph module
	:rtype: torch.fx.GraphModule
	"""
	leaves = []
	traceable = False
	while not traceable:
		tracer = PatchedTracer(leaves)
		finder = NonTraceableFinder(tracer)
		finder.get_non_traceable_modules(model, sample)
		new_leaves = finder.get_non_traceable_leaf_modules(model)
		assert all([leaf not in leaves for leaf in new_leaves]), (
			"New leaves should not be in already fixed leaves: "
			+ str([leaf for leaf in new_leaves if leaf in leaves])
		)
		traceable = len(new_leaves) == 0
		leaves.extend(new_leaves)

	graph = tracer.trace(model, concrete_args={"dummy_run": False, "return_latent": False})
	return torch.fx.GraphModule(model, graph)


def extract_graph(model, sample, mode="fx"):
	"""
	Extract graph using specified mode.

	:param model: Model to extract graph from
	:type model: nn.Module
	:param sample: Sample input to use for tracing
	:type sample: torch.Tensor
	:param mode: Graph extraction mode ('export', 'fx', or 'fx_safe')
	:type mode: str
	:return: Graph module
	:rtype: torch.fx.GraphModule or torch.export.ExportedProgram
	:raises ValueError: If mode is not recognized
	"""
	if mode == "export":
		return extract_graph_export(model, sample)
	elif mode == "fx":
		return extract_graph_fx(model)
	elif mode == "fx_safe":
		return extract_graph_fx_safe(model, sample)
	else:
		raise ValueError(f"Unknown graph extraction mode: {mode}")


def try_extract_graph(model, sample):
	"""
	Try extracting graph using different modes.

	:param model: Model to extract graph from
	:type model: nn.Module
	:param sample: Sample input to use for tracing
	:type sample: torch.Tensor
	:return: Graph module
	:rtype: torch.fx.GraphModule or torch.export.ExportedProgram
	:raises Exception: If all extraction modes fail
	"""
	modes = ["fx", "fx_safe", "export"]
	for mode in modes:
		try:
			return extract_graph(model, sample, mode)
		except Exception as e:
			logger.debug(str(e))
			logger.info(f"Failed to extract graph with mode {mode}. Trying next mode.")
	raise Exception("Failed to extract graph")


class PatchedTracer(LayerDWTracer):
	"""
	Extended tracer that handles additional leaf modules.
	"""

	def __init__(self, fx_leaf_modules=[]):
		super().__init__()
		self.fx_leaf_modules = fx_leaf_modules

	def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
		if m.__class__ in self.fx_leaf_modules:
			return True

		return super().is_leaf_module(m, module_qualified_name)


class NonTraceableFinder:
	"""
	Utility class to find non-traceable modules in a model
	"""

	def __init__(self, tracer):
		self.non_traceable_modules = []
		self.non_traceable_leaf_modules = []
		self.fx_tracer = tracer

	def get_non_traceable_modules(self, model, args):
		"""
		Find non-traceable modules in model.

		:param model: Model to analyze
		:type model: nn.Module
		:param args: Sample inputs for model
		:type args: tuple
		:return: List of non-traceable module names
		:rtype: list
		"""
		self.non_traceable_modules = []
		for name, module in model.named_modules():
			if isinstance(module, torch.nn.ModuleList) or name == "":
				continue
			if self.fx_tracer.is_leaf_module(module, name):
				continue
			try:
				self.fx_tracer.trace(module)
			except Exception:
				self.non_traceable_modules.append(name)

		self.non_traceable_modules = list(set(self.non_traceable_modules))
		return self.non_traceable_modules

	def get_non_traceable_leaf_modules(self, model):
		"""
		Get leaf modules that are non-traceable.

		:return: List of non-traceable leaf module names
		:rtype: list
		"""
		filtered_paths = filter_paths(self.non_traceable_modules)
		self.non_traceable_leaf_modules = list(
			set([extract_module_by_name(model, path).__class__ for path in filtered_paths])
		)
		return self.non_traceable_leaf_modules


def replace_module_inplace(root_module, module_name, new_module):
	# Split the hierarchical name into components
	module_parts = module_name.split(".")
	current_module = root_module

	# Traverse to the parent of the target module
	for part in module_parts[:-1]:
		if not hasattr(current_module, part):
			raise ValueError(f"Module '{part}' not found in '{current_module.__class__.__name__}'.")
		current_module = getattr(current_module, part)

	# Extract the target module
	target_name = module_parts[-1]
	if not hasattr(current_module, target_name):
		raise ValueError(f"Module '{target_name}' not found in '{current_module.__class__.__name__}'.")
	replaced_module = getattr(current_module, target_name)

	# Replace the target module in-place
	setattr(current_module, target_name, new_module)

	return replaced_module


def extract_module_by_name(root_module, module_name):
	"""
	Recursively extract a submodule by its hierarchical name.

	:param root_module: The root module to start searching from
	:type root_module: nn.Module
	:param module_name: Hierarchical name of the submodule (e.g., "submodule1.layer")
	:type module_name: str
	:return: The extracted submodule
	:rtype: nn.Module
	:raises ValueError: If the specified submodule is not found
	"""
	module_parts = module_name.split(".")
	current_module = root_module

	for part in module_parts:
		if hasattr(current_module, part):
			current_module = getattr(current_module, part)
		else:
			if isinstance(current_module, nn.ModuleList):
				try:
					current_module = current_module[int(part)]
				except Exception as _:
					raise ValueError(f"Module '{part}' not found in '{current_module.__class__.__name__}'.")

	return current_module


def path_to_module(root, module):
	"""
	Extract the name of a module in a model.
	"""
	for name, child in root.named_children():
		if child is module:
			return name
	raise NameError(f"Module {module}, type {type(module)}, not found in {root}, type {type(root)}")


def filter_paths(input_paths):
	"""
	Filter paths to keep only the leaves.

	:param input_paths: List of module paths
	:type input_paths: list
	:return: Set of filtered paths
	:rtype: set
	"""
	# Step 1: Sort the paths by length
	sorted_paths = sorted(input_paths, key=lambda p: len(p.strip().split(".")), reverse=True)

	# Step 2: Initialize an empty set for output paths
	output_paths = set()

	# Step 3: Iterate through sorted paths
	for path in sorted_paths:
		# Check if path is a subpath of any existing path in output_paths
		is_subpath = any(existing_path.startswith(path + ".") for existing_path in output_paths)

		# If not a subpath, add to output paths
		if not is_subpath:
			output_paths.add(path)

	return output_paths


def get_shapes(input):
	"""
	Get shapes of input tensors recursively.

	:param input: Input tensor or container of tensors
	:type input: torch.Tensor or list or tuple or dict
	:return: Shape(s) of input tensor(s)
	:rtype: tuple or list or dict
	"""
	if isinstance(input, torch.Tensor):
		return tuple(input.shape)
	elif isinstance(input, (list, tuple)):
		return type(input)(get_shapes(item) for item in input)
	elif isinstance(input, dict):
		return {key: get_shapes(value) for key, value in input.items()}
	else:
		# For non-tensor and non-container types, return as is
		return input
