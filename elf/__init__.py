"""
ELF: memory-aware pipeline parallelism for PyTorch.

This module re-exports everything needed for the documented workflows: building a pipeline,
describing how it is partitioned, writing or inspecting a schedule, plugging in a custom algorithm,
and driving the execution by hand. Anything not listed in ``__all__`` is reachable through its own
module (for instance ``elf.scheduling.ilp`` for the rematerialisation ILP, or
``elf.execution.remat``), but is not part of the surface this package promises to keep stable.
"""

from elf.pipeline import Pipeline, PipelineConfig
from elf.partitioners import PartitionResult, partition
from elf.profiling import StageProfile, profile_pipeline
from elf.signature import Signature, sequential_signatures
from elf.topology import Placement
from elf.utils import TensorMetadata, Timer
from elf import flops  # noqa: F401 - registering the FLOP formulas is the point
from elf.scheduling import (
	Operation,
	OperationType,
	OpOptions,
	RematMode,
	FixedSchedule,
	add_comms,
	schedule_to_str,
	check_schedule_validity,
)
from elf.registry import (
	COMM_SCHEDULERS,
	PARTITIONERS,
	SCHEDULERS,
	TRACERS,
	CommScheduler,
	Partitioner,
	Registry,
	Scheduler,
	Tracer,
	resolve,
)
from elf.zb_utils import replace_layer_with_layer_dw, replace_linear_with_linear_dw

__all__ = [
	# Entry point
	"Pipeline",
	"PipelineConfig",
	# Describing the pipeline
	"Placement",
	"Signature",
	"sequential_signatures",
	"partition",
	"PartitionResult",
	# Measuring what the stages cost
	"StageProfile",
	"profile_pipeline",
	# Writing and inspecting a schedule
	"Operation",
	"OperationType",
	"OpOptions",
	"RematMode",
	"FixedSchedule",
	"add_comms",
	"schedule_to_str",
	"check_schedule_validity",
	# Plugging in an algorithm
	"SCHEDULERS",
	"COMM_SCHEDULERS",
	"PARTITIONERS",
	"TRACERS",
	"Registry",
	"resolve",
	"Scheduler",
	"CommScheduler",
	"Partitioner",
	"Tracer",
	# Instrumentation and the point-to-point wire format
	"Timer",
	"TensorMetadata",
	# Zero-bubble layers patching
	"replace_layer_with_layer_dw",
	"replace_linear_with_linear_dw",
]
