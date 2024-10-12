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