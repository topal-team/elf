"""
Automatic partition of a computation graph
"""

from .tracing import try_extract_graph, extract_graph
from .partition import partition_graph
from .profile import profile_operations
from .utils import Signature, signatures_from_sources_targets, get_sources_targets_sequential

from .custom import split_graph, split_graph_constrained
from .metis import split_graph_metis
from .dagP import split_graph_dagP

from ..registry import PARTITIONERS

PARTITIONERS.register(
	"naive",
	split_graph,
	"Naively partition into roughly equal blocks without taking memory into account.",
)
PARTITIONERS.register(
	"constrained",
	split_graph_constrained,
	"Naively partition into roughly equal blocks, with a constraint of 1 input and 1 output per block.",
)
PARTITIONERS.register(
	"metis", split_graph_metis, "Graph partitioning using METIS. Requires gpmetis to be installed."
)
PARTITIONERS.register(
	"dagP", split_graph_dagP, "Graph partitioning using dagP. Requires rMLGP to be installed."
)

__all__ = [
	"try_extract_graph",
	"extract_graph",
	"partition_graph",
	"profile_operations",
	"Signature",
	"signatures_from_sources_targets",
	"get_sources_targets_sequential",
]
