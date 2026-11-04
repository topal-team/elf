"""
Scheduling strategies for both computations and communications.
"""

from elf.scheduling.scheduling import (
	Operation,
	OperationType,
	OpOptions,
	RematMode,
	comm_types,
	compute_types,
	matching,
	get_peer_rank,
	schedule_to_str,
	check_schedule_validity,
	mark_replayable_forwards,
)
from elf.scheduling.comm_scheduling import *
from elf.scheduling.schedulers import *
from elf.topology import is_cyclic, is_mirror, is_single_wave

from elf.registry import SCHEDULERS, COMM_SCHEDULERS, Scheduler


SCHEDULERS.register(
	["afab", "gpipe"],
	Scheduler(generate_afab_schedule, is_single_wave),  # all placements are valid
	"All Forward All Backward as in GPipe https://arxiv.org/abs/1811.06965",
)

SCHEDULERS.register(
	["1f1b", "1F1B", "megatron"],
	Scheduler(
		generate_1f1b_schedule, is_cyclic
	),  # single and multiwave supported, but only cyclic stage assignment
	"One Forward One Backward as in PipeDream https://arxiv.org/abs/1806.03377",
)

SCHEDULERS.register(
	"hanayo",
	Scheduler(
		generate_hanayo_schedule, is_mirror
	),  # is_mirror cannot be true on singlewave schedules, so no need to add not(is_single_wave)
	"Hanayo schedule as in https://dl.acm.org/doi/10.1145/3581784.3607073",
)

SCHEDULERS.register(
	"full_remat",
	Scheduler(generate_full_remat_schedule, is_single_wave),  # only check on 1-wave
	"Efficient scheduling with rematerialization of everything. Useful for memory-constrained setups.",
)


SCHEDULERS.register(
	"zbh1",
	Scheduler(generate_zbh1_schedule, is_single_wave),
	"ZB-H1 schedule as in https://arxiv.org/abs/2401.10241",
)


SCHEDULERS.register(
	"zbh2",
	Scheduler(generate_zbh2_schedule, is_single_wave),
	"ZB-H2 schedule as in https://arxiv.org/abs/2401.10241",
)


SCHEDULERS.register(
	"zbv",
	Scheduler(generate_zbv_schedule, is_mirror),
	"ZB-V schedule as in https://arxiv.org/abs/2401.10241. Warning: only works with 2 stages per device.",
)

SCHEDULERS.register(
	"inference", generate_inference_schedule, "Inference schedule. Forward pass only."
)


SCHEDULERS.register(
	"inference", generate_inference_schedule, "Inference schedule. Forward pass only."
)

SCHEDULERS.register(
	"fixed", FixedSchedule, "Fixed schedule. Use a dictionary to specify the schedule."
)

SCHEDULERS.register(
	"auto",
	lambda *_: None,
	"Auto scheduler that solves the ILP for all base schedulers and uses the best objective value.",
)  # dummy scheduler: it will be overridden by the AutoScheduler class during stage-rematification
# this trick is a bit ugly, I'd rather have a proper registration that solves lazily

COMM_SCHEDULERS.register(
	"topo",
	topological_sort,
	"Topological sort of the communications. No optimizations are performed.",
)

COMM_SCHEDULERS.register(
	"simulation",
	smart_topological_sort,
	"Simulate execution to schedule communications together across ranks.",
)

# COMM_SCHEDULERS.register(
# 	"pipelined",
# 	pipelined_topological_sort,
# 	"Pipelines receive -> compute -> send to utilize compute/comm overlapping.",
# )

COMM_SCHEDULERS.register(
	"overlap",
	overlap_comm_scheduler,
	"Post RECV as soon as possible to completely overlap it. Simulates faster than 'safe-overlap',"
	" but CAN DEADLOCK: it posts RECVs earlier than is safe once a computation synchronizes the"
	" device. Prefer 'safe-overlap' unless you have measured the difference and know that no"
	" computation in your model ever synchronizes.",
)

COMM_SCHEDULERS.register(
	"safe-overlap",
	safe_overlap_comm_scheduler,
	"Post every RECV at the earliest slot that provably cannot deadlock.",
)

COMM_SCHEDULERS.register(
	"timed-overlap",
	timed_overlap_comm_scheduler,
	"Post every RECV just in time: as late as the measured timeline allows without delaying its"
	" transfer, and never so early that it cannot be live. Needs the stage profiles, which the"
	" pipeline measures at setup unless profiling is turned off; without them it falls back to"
	" 'safe-overlap'.",
)

COMM_SCHEDULERS.register(
	"toporecv",
	toposort_with_recv_prio_comm_scheduler,
	"Topological sort with priority on RECV > op that frees a RECV > everything else",
)


__all__ = [
	# Schedule vocabulary
	"Operation",
	"OperationType",
	"OpOptions",
	"RematMode",
	"comm_types",
	"compute_types",
	"matching",
	"get_peer_rank",
	"schedule_to_str",
	"check_schedule_validity",
	"mark_replayable_forwards",
	# Schedule construction
	"add_comms",
	# Communication scheduling
	"DirectedGraph",
	"schedule_to_graph",
	"recv_types",
	"send_types",
	# pre-registered functions are accessible through the registries
]
