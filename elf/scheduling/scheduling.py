"""
Manipulate dependency graphs corresponding to schedules
"""

from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
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
]


class OperationType(Enum):
	"""
	Different type of operations that can be performed.
	They can be both computation (forward, backward, ..) or communications (p2p send/recv, offloading, ..)
	"""

	RECV_FORWARD = 0
	FORWARD = 1
	SEND_FORWARD = 2
	RECV_BACKWARD = 3
	BACKWARD_INPUTS = 4
	BACKWARD_PARAMS = 5
	SEND_BACKWARD = 6
	LOSS_FORWARD = 7
	LOSS_BACKWARD = 8
	ALL_REDUCE_PARAM_GRADS = 9
	RECOMPUTE_FORWARD = 10  # between F and B, for a forward that kept nothing
	RECOMPUTE_BACKWARD_INPUTS = 11
	PREFETCH_ACTIVATIONS = 12
	RECOMPUTE_FORWARD_FOR_PARAMS = 13  # between B and W, for activations dropped after B

	def __repr__(self) -> str:
		return self.name.lower()

	def __str__(self) -> str:
		return self.name.lower()


comm_types = {
	OperationType.RECV_FORWARD,
	OperationType.RECV_BACKWARD,
	OperationType.SEND_FORWARD,
	OperationType.SEND_BACKWARD,
}


# Note: if you want to include "Loss Forward" and "Loss Backward" in the compute types, check the smart toposorting algorithm; those 2 specially need to be skipped
compute_types = {
	OperationType.FORWARD,
	OperationType.BACKWARD_INPUTS,
	OperationType.BACKWARD_PARAMS,
	OperationType.RECOMPUTE_FORWARD,
	OperationType.RECOMPUTE_FORWARD_FOR_PARAMS,
	OperationType.RECOMPUTE_BACKWARD_INPUTS,
}


class OpOptions(StrEnum):  # will be used as a key in a dict, needs to be a string
	"""
	Options that can be passed to operations to modify their behaviour.

	Attributes
	----------
	REMAT_STRATEGY : str
		Remat strategy is a function that tells, for each module, how its activations are handled.
		Signature: ``(name: str, module: nn.Module) -> RematMode | None``, where None leaves the
		module alone.
	RECOMPUTE_ACTIVATIONS : str
		Recompute Backward-Forward (activations, between B and W).
	RECOMPUTE_GRADIENTS : str
		Recompute Backward-Backward (gradients, between B and W).
	ACTIVATION_OFFLOAD : str
		Offload activations to CPU.
	REPLAYABLE : str
		Set on a forward whose activations a later operation rebuilds, so that it keeps what the
		replay starts from. Filled in by :func:`mark_replayable_forwards`, not by the schedulers.
	"""

	REMAT_STRATEGY = auto()
	RECOMPUTE_ACTIVATIONS = auto()
	RECOMPUTE_GRADIENTS = auto()
	ACTIVATION_OFFLOAD = auto()
	REPLAYABLE = auto()

	def __repr__(self) -> str:
		return str(self)


class RematMode(Enum):
	"""
	What a ``REMAT_STRATEGY`` asks of a module.

	Attributes
	----------
	KEEP
		Keep the activations until a schedule drops them. Only useful in a stage whose activations
		are replayed between ``B`` and ``W``, where every module is covered by default.
	CHECKPOINT
		Recompute the activations during the backward, from inputs the module always keeps, as
		:func:`torch.utils.checkpoint.checkpoint` does. What it returns is still rebuilt rather
		than kept, unless another module keeps it as its own input.
	DISCARD
		Recompute the activations, and keep the inputs only if no recomputed module can rebuild
		them. Reading the output of another recomputed module re-runs that module during the
		backward, so this only pays when that module is cheap, a pointwise one say.
	"""

	KEEP = auto()
	CHECKPOINT = auto()
	DISCARD = auto()

	def __repr__(self) -> str:
		return f"RematMode.{self.name}"


@dataclass(frozen=True)
class Operation:
	"""
	Computation or communication unit. Operations are the elements contained in the schedule, and give all the informations needed for a block to execute it, except the data itself.
	"""

	block_id: int
	mb_id: int | None  # some ops do not operate on a specific mb
	op: OperationType
	rank: int
	peer: int | None = None  # src / dst of communications
	options: dict[OpOptions, Any] = field(
		default_factory=dict, compare=False
	)  # options are not taken into account when comparing or hashing. That's fine because 2 ops with the same
	# (op, mb_id, block_id, peer) never coexist in the same schedule, see check_schedule_validity

	def __str__(self) -> str:
		return f"[Block {self.block_id}, mb {self.mb_id}]:{repr(self.op)}({self.options})"

	def __repr__(self) -> str:
		return str(self)


matching_ops = {
	OperationType.RECV_FORWARD: OperationType.SEND_FORWARD,
	OperationType.RECV_BACKWARD: OperationType.SEND_BACKWARD,
	OperationType.SEND_FORWARD: OperationType.RECV_FORWARD,
	OperationType.SEND_BACKWARD: OperationType.RECV_BACKWARD,
}


def matching(op_type):
	"""
	The matching operation is the opposite communication, in the same direction.
	For instance, a recv(forward) is matched with a send(forward), and a send(backward) with a recv(backward).
	Returns None for any operation that is not a communication.
	"""
	return matching_ops.get(op_type, None)


def get_peer_rank(op, placement):
	"""
	Get the peer rank of a communication
	"""
	if op.peer is None:
		return None
	return placement[op.peer]


def schedule_to_str(schedule, print_comms=False):
	"""
	Convert a schedule to a string representation.

	:param schedule: Schedule to convert
	:type schedule: List[Operation]
	:param print_comms: Whether to include communications
	:type print_comms: bool
	:return: String representation of the schedule
	:rtype: str
	"""
	reprs = {
		OperationType.RECV_FORWARD: "rf",
		OperationType.FORWARD: "f",
		OperationType.SEND_FORWARD: "sf",
		OperationType.RECV_BACKWARD: "rb",
		OperationType.BACKWARD_INPUTS: "b",
		OperationType.BACKWARD_PARAMS: "w",
		OperationType.SEND_BACKWARD: "sb",
		OperationType.RECOMPUTE_FORWARD: "R",
		OperationType.RECOMPUTE_FORWARD_FOR_PARAMS: "Rw",
		OperationType.RECOMPUTE_BACKWARD_INPUTS: "R*",
		OperationType.ALL_REDUCE_PARAM_GRADS: "(AR)",
		OperationType.PREFETCH_ACTIVATIONS: "P",
	}

	def shorten(op):
		letter = reprs[op.op]
		if op.op == OperationType.FORWARD:
			letter = "F" if op.options.get(OpOptions.REMAT_STRATEGY) else "f"
		elif op.op == OperationType.BACKWARD_INPUTS:
			letter = "B" if op.options.get(OpOptions.RECOMPUTE_ACTIVATIONS, False) else "b"
		return letter

	ranks = sorted(set(op.rank for op in schedule))
	lines = []
	for rank in ranks:
		rank_ops = [
			op
			for op in schedule
			if op.rank == rank and op.op in reprs and (print_comms or op.op not in comm_types)
		]
		ops_str = " ".join(
			filter(
				lambda s: s != "",
				[
					f"{shorten(op)}{op.block_id}-{op.mb_id if op.mb_id is not None else ''}"
					+ (f"({op.peer})" if op.op in comm_types else "")
					for op in rank_ops
				],
			)
		)
		lines.append(f"Rank {rank}: {ops_str}")
	return "\n".join(lines)


def mark_replayable_forwards(schedule):
	"""
	Tell each forward whether anything will rebuild the activations it produces.

	Rebuilding them means replaying the forward, which only works if it kept what the replay starts
	from. Whether that happens is decided by the schedule, not by the forward, so it has to be
	carried back to it before the schedule runs.

	The condition is that a recompute-forward exists for the same block and micro-batch, in either
	of the places one is scheduled: ``RECOMPUTE_FORWARD`` between ``F`` and ``B`` for a stage that
	recomputes itself, and ``RECOMPUTE_FORWARD_FOR_PARAMS`` between ``B`` and ``W`` for one whose
	activations are dropped after its backward.

	:param schedule: operations of the whole schedule, in order
	:type schedule: List[Operation]
	:return: the same operations, with ``REPLAYABLE`` set on the forwards concerned
	:rtype: List[Operation]
	"""
	replayed = {
		(op.block_id, op.mb_id)
		for op in schedule
		if op.op in (OperationType.RECOMPUTE_FORWARD, OperationType.RECOMPUTE_FORWARD_FOR_PARAMS)
	}

	for op in schedule:
		if op.op is OperationType.FORWARD and (op.block_id, op.mb_id) in replayed:
			op.options[OpOptions.REPLAYABLE] = True

	return schedule


def check_schedule_validity(schedule):
	"""
	Perform basic sanity checks on a generated schedule.

	.. warning::
		Some of these checks may not pass for your custom, specific schedule. If this is the case, please report it to the developers.
	"""
	seen = set()
	for op in schedule:
		assert op not in seen, f"{op} appears twice in the schedule"
		seen.add(op)

	stages = set(op.block_id for op in schedule)
	n_mb = len(set(op.mb_id for op in schedule if op.mb_id is not None))

	op_indices = {}
	for idx, op in enumerate(schedule):
		if (op.block_id, op.mb_id, op.op) not in op_indices:
			op_indices[op.block_id, op.mb_id, op.op] = idx

	for stage in stages:
		rank_ops = [op for op in schedule if op.block_id == stage]
		rank_mb_ids = set(op.mb_id for op in rank_ops if op.mb_id is not None)
		assert len(rank_mb_ids) == n_mb, (
			f"[Stage {stage}] has {len(rank_mb_ids)} microbatches, expected {n_mb}"
		)

		for mb_id in rank_mb_ids:
			f = op_indices.get((stage, mb_id, OperationType.FORWARD), None)
			# in case of no backward, we allow to have 0 forward without remat
			b = op_indices.get((stage, mb_id, OperationType.BACKWARD_INPUTS), None)
			assert f is not None or b is None, (
				f"[Stage {stage}] Forward should be present for microbatch {mb_id}"
			)
			assert b is None or f < b, (
				f"[Stage {stage}] Forward should be before backward inputs for microbatch {mb_id}"
			)  # we allow "no backward"
			w = op_indices.get((stage, mb_id, OperationType.BACKWARD_PARAMS), None)
			assert w is not None or b is None, (
				f"[Stage {stage}] Backward params should be present for microbatch {mb_id} since backward inputs is present"
			)
			assert w is None or b < w, (
				f"[Stage {stage}] Backward params should be after backward inputs for microbatch {mb_id}"
			)
			rf = op_indices.get((stage, mb_id, OperationType.RECOMPUTE_FORWARD), None)
			assert rf is None or (f is not None and b is not None and f < rf < b), (
				f"[Stage {stage}] Recompute forward should be between forward and backward inputs for microbatch {mb_id}"
			)
			rfp = op_indices.get((stage, mb_id, OperationType.RECOMPUTE_FORWARD_FOR_PARAMS), None)
			assert rfp is None or (b is not None and w is not None and b < rfp < w), (
				f"[Stage {stage}] Recompute forward for params should be between backward inputs and backward params for microbatch {mb_id}"
			)
			rb = op_indices.get((stage, mb_id, OperationType.RECOMPUTE_BACKWARD_INPUTS), None)
			assert rb is None or (rfp is not None and rfp < rb < w), (
				f"[Stage {stage}] Recompute backward inputs should be between recompute forward for params and backward params for microbatch {mb_id}"
			)
