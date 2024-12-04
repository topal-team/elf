"""
Utils to partition a computation graph using dagP
"""

import os
import shutil
import tempfile
import subprocess
from .metis import read_metis, convert_fx


def split_graph_dagP(graph, times, memories, n):
	"""
	Split graph using dagP (https://github.com/GT-TDAlab/dagP, https://inria.hal.science/hal-02306566/document)
	Compared to METIS, this has the advantage of always creating acyclic graphs

	:param graph: symbolic trace of the module to partition (see torch.fx)
	:type graph: fx.GraphModule
	:param times: information about the weights, i.e. the profiled execution time, of each node
	:type times: Dict[str, float]
	:param memories: information about the weight of the outgoing edge for each node, i.e. the memory size of the output value
	:type memories: Dict[str, float]
	:param n: number of partitions to create
	:type n: Optional[int]

	:return: ``n`` lists of nodes corresponding to each part
	:rtype: List[List[fx.Node]]
	"""
	assert shutil.which(
		"rMLGP"
	), "dagP chosen as partition strategy, but can't find it. Please make sure it is installed and findable in the PATH."
	graph = convert_fx(graph, times, memories)
	file = write_dagP(graph)
	execute_dagP(file, n)
	file.close()
	parts = read_metis(
		graph, f"{file.name}.partsfile.part_{n}.seed_0.txt"
	)  # output file is the same as metis

	return parts


def write_dagP(graph):
	"""
	Dumps the info about a custom graph into a file in DOT format

	:param graph: custom representation of every node of a computational graph, by name
	:type graph: Dict[str, Node]

	:rtype: file descriptor where the info was written
	:type: File
	"""
	file = tempfile.NamedTemporaryFile("w+", dir=".", suffix=".dot")
	file.write("digraph compgraph {\n")  # don't forget the \n because dagP dot reader is sensitive ;)
	for node in graph.values():
		file.write(node.to_dot_line())
	for node in graph.values():
		file.write(node.to_dot_edges())
	file.write("}")
	return file


def execute_dagP(file, n):
	"""
	Execute dagP on a file and clean all unnecessary files

	:param file: file descriptor of the file where the info of the graph was written
	:type file: File
	:param n: number of parts to create
	:type n: int
	"""
	file.flush()  # Ensure all data is written before executing
	file.seek(0)  # Reset file pointer to the beginning for reading in subprocess
	subprocess.run(
		[
			"rMLGP",
			file.name,
			str(n),
			"--obj",
			"1",
			"--write_parts",
			"1",
			"--print",
			"0",
			"--ratio",
			"1.2",
			"--runs",
			"10",
		],
		stdout=subprocess.DEVNULL,
	)
	os.remove(f"{file.name}.bin")
	os.remove(f"{file.name}.nodemappings")
	os.remove(f"{file.name}.partitioned.part_{n}.seed_0.dot")


# No read_dagP because the output is the same as METIS. Use read_metis instead.
