"""
Utils for operation profiling
"""

import torch
import numpy as np
from pipeline.utils import Timer

DONT_CUT_HERE = 2 << 40


class Profiler(torch.fx.Interpreter):
	def __init__(self, niter=10, *args, **kwargs):
		self.times = {}
		self.memories = {}
		self.niter = niter
		self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
		super(Profiler, self).__init__(*args, **kwargs)

	def to_device(self, x, device):
		"""
		Moves ``x`` to the specified device if it is a tensor
		If ``x`` is an iterable or dictionary, recursively moves every tensor contained in it to the device too
		If ``x`` is a Node, modifies its value directly in the Interpreter's environment
		Otherwise does not do anything

		:param x: object to move to device memory
		:type x: Any
		:param device: destination
		:type device: str or torch.device

		:return: moved objects
		:rtype: same as ``x``
		"""
		if isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Module):
			return x.to(device, non_blocking=True)

		elif isinstance(x, torch.fx.Node):
			self.env[x] = self.to_device(self.env[x], device)
			return x

		elif isinstance(x, dict):
			return {k: self.to_device(v, device) for k, v in x.items()}

		elif hasattr(x, "__iter__") and not isinstance(x, (str, torch.fx.proxy.Proxy)):
			return type(x)(self.to_device(item, device) for item in x)

		return x

	def move_dependencies(self, node, device):
		"""
		Moves all of the node's arguments to the specified device.

		:param node: The node whose dependencies need to be moved
		:type node: torch.fx.Node
		:param device: The target device to move dependencies to
		:type device: str or torch.device
		"""
		for i in range(len(node.args)):
			node.update_arg(i, self.to_device(node.args[i], device))
		for key in node.kwargs.keys():
			node.update_kwarg(key, self.to_device(node.kwargs[key], device))

		# Special case: modules are not in args/kwargs but in target.
		if node.op == "call_module":
			self.to_device(self.fetch_attr(node.target), device)

	def run_node(self, node):
		# Move all inputs to the specified device
		# self.env is an internal from Interpreter; it's hacky to modifiy it
		self.move_dependencies(node, self.device)

		if torch.cuda.is_available():
			torch.cuda.synchronize()

		times = []
		for _ in range(self.niter):
			with Timer() as timer:
				result = super().run_node(node)
			times.append(timer.time())

		result = self.to_device(result, "cpu")

		self.move_dependencies(node, "cpu")

		self.times[node.name] = np.median(times)
		self.memories[node.name] = get_memory(result)

		return result


def get_memory(x):
	"""
	Estimates memory used by ``x`` if it is a tensor
	If ``x`` is an iterable or dictionary, recursively count memory for every tensor contained in it

	:param x: object to estimate
	:type x: Any

	:return: memory estimation in bytes
	:rtype: float
	"""
	if isinstance(x, torch.Tensor):
		return x.numel() * x.element_size()
	# if isinstance(x, torch.nn.Parameter):
	# 	return DONT_CUT_HERE
	# elif isinstance(x, dict):
	# 	return sum([get_memory(v) for v in x.values()])
	# elif hasattr(x, "__iter__") and not isinstance(x, (str, torch.fx.proxy.Proxy)):
	# 	return sum([get_memory(v) for v in x])
	# return 0

	# We dont want to cut anywhere that is not a tensor
	return DONT_CUT_HERE


def profile_operations(graph_module, input_sample, niter=10):
	"""
	Get time and memory for each node of a traced module, when running forward on a sample.

	:param graph_module: original symbolically traced module to be profiled (see torch.fx)
	:type graph_module: fx.GraphModule
	:param input_sample: example of inputs to feed to the model
	:type input_sample: Tensor
	:param niter: number of times each operation will be profiled
	:type niter: int

	:return: 2 dicts, one for time and one for memory used, respectively. Each one has the format {node_name: value}
	:rtype: fx.GraphModule, Dict[str, List[float]], Dict[str, List[float]]
	"""
	# Save original buffers
	original_buffers = {}
	for name, buffer in graph_module.named_buffers():
		original_buffers[name] = buffer.clone().detach()
		
	graph_module.train()
	profiler = Profiler(niter, graph_module)
	with torch.no_grad():
		profiler.boxed_run([input_sample])

	# Restore original buffers
	for name, buffer in graph_module.named_buffers():
		buffer.data.copy_(original_buffers[name])

	# With boxed runs placeholders bypass the profiler, we need to manually add them afterwards
	for node in graph_module.graph.nodes:
		if node.op == "placeholder":
			profiler.memories[node.name] = get_memory(input_sample)  # TODO: handle multiple inputs
			profiler.times[node.name] = 0
		if node.op == "output":
			profiler.times[node.name] = 0

	return profiler.times, profiler.memories
