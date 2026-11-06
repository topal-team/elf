"""
Automatic partition of a computation graph
"""

from elf.partitioners.tracing import (
	try_extract_graph,
	extract_graph_fx,
	extract_graph_fx_safe,
	extract_graph_export,
	trace,
)
from elf.partitioners.partition import PartitionResult, partition, split
from elf.partitioners.profile import profile_operations

from elf.partitioners.custom import split_graph, split_graph_balanced
from elf.partitioners.metis import split_graph_metis
from elf.partitioners.dagP import split_graph_dagP

from elf.registry import PARTITIONERS, TRACERS

PARTITIONERS.register(
	"naive",
	split_graph,
	"Naively partition into roughly equal blocks without taking memory into account.",
)
PARTITIONERS.register(
	"balanced",
	split_graph_balanced,
	"Partition into contiguous blocks, searching for the best time balance with few tensors per cut.",
)
PARTITIONERS.register(
	"metis", split_graph_metis, "Graph partitioning using METIS. Requires gpmetis to be installed."
)
PARTITIONERS.register(
	"dagP", split_graph_dagP, "Graph partitioning using dagP. Requires rMLGP to be installed."
)

TRACERS.register("default", try_extract_graph, "Try all tracers and use the first one that works")
TRACERS.register("fx", extract_graph_fx, "Extract graph using torch.fx")
TRACERS.register(
	"fx_safe",
	extract_graph_fx_safe,
	"Extract graph using torch.fx, by marking every non-traceable submodule as leaf, iteratively.",
)
TRACERS.register("export", extract_graph_export, "Extract graph using torch.export")


__all__ = ["PartitionResult", "partition", "trace", "split", "profile_operations"]
