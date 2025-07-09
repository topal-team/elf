from .scheduling import schedule_to_str, check_schedule_validity
from .comm_scheduling import reorder_communications
from .schedulers import *

from ..registry import SCHEDULERS


SCHEDULERS.register(
	["afab", "gpipe"],
	generate_afab_schedule,
	"All Forward All Backward as in GPipe https://arxiv.org/abs/1811.06965",
)

SCHEDULERS.register(
	["1f1b", "megatron"],
	generate_1f1b_schedule,
	"One Forward One Backward as in PipeDream https://arxiv.org/abs/1806.03377",
)

SCHEDULERS.register(
	"hanayo",
	generate_hanayo_schedule,
	"Hanayo schedule as in https://dl.acm.org/doi/10.1145/3581784.3607073",
)

SCHEDULERS.register(
	"full_remat",
	generate_full_remat_schedule,
	"Efficient scheduling with rematerialization of everything. Useful for memory-constrained setups.",
)


SCHEDULERS.register(
	"zbh1", generate_zbh1_schedule, "ZB-H1 schedule as in https://arxiv.org/abs/2401.10241"
)


SCHEDULERS.register(
	"zbh2", generate_zbh2_schedule, "ZB-H2 schedule as in https://arxiv.org/abs/2401.10241"
)


SCHEDULERS.register(
	"zbv",
	generate_zbv_schedule,
	"ZB-V schedule as in https://arxiv.org/abs/2401.10241. Warning: only tested with 4 processors, 8 micro batches, 2 stages per device.",
)

SCHEDULERS.register(
	"inference", generate_inference_schedule, "Inference schedule. Forward pass only."
)


__all__ = ["schedule_to_str", "check_schedule_validity", "reorder_communications", "JsonScheduler"]
