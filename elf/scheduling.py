"""
Manipulate dependency graphs corresponding to schedules
"""

from enum import Enum, StrEnum, auto

import logging
import torch.distributed as dist

logger = logging.getLogger("scheduling")


class OperationType(Enum):
	"""
	Different type of operations that can be performed.
	They can be both computation (forward, backward, ..) or communications (p2p send/recv, ..)
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
	RECOMPUTE_FORWARD = 10
	RECOMPUTE_BACKWARD_INPUTS = 11

	def __repr__(self) -> str:
		return self.name.lower()

	def __str__(self) -> str:
		return self.name.lower()


class OpOptions(StrEnum):  # will be used as a key in a dict, needs to be a string
	"""
	Options that can be passed to operations to modify their behaviour
	"""

	REMAT_STRATEGY = auto()

	# SAVE is used differently depending on the operation type
	# for forward, it's a boolean to save the activations or not
	# for backward, it's a string among ["full", "gradients", "none"]. "full" means keeping the partial activations and gradients, "gradients" means only the gradients, "none" means deleting both.
	SAVE = auto()
	
	BATCHED_COMM = auto()
	OFFLOAD_DW = auto()


class Operation:
	"""
	Computation or communication unit. Operations are the elements contained in the schedule, and give all the informations needed for a block to execute it, except the data itself.
	"""

	def __init__(self, block_id, mb_id, op, rank, **options):
		"""
		:param block_id: number of the block that will execute this OP in the pipeline
		:type block_id: int
		:param mb_id: number of the micro batch that this op will be executed on
		:type mb_id: int
		:param op: type of operation
		:type op: OperationType
		:param rank: global rank of the block
		:type rank: int
		:param **options: options to modify the behaviour of the execution
		"""
		self.block_id = block_id
		self.op = op  # type of the operation (see OperationType enum)
		self.mb_id = mb_id  # micro batch id in the batch
		self.rank = rank
		self.options = options

	def __str__(self) -> str:
		return f"[Block {self.block_id}, mb {self.mb_id}]:{repr(self.op)}({self.options})"

	def __repr__(self) -> str:
		return str(self)

	def __eq__(self, __value: object) -> bool:
		return (
			__value.op == self.op
			and __value.mb_id == self.mb_id
			and __value.rank == self.rank
			and __value.options == self.options
		)

	def __hash__(self) -> int:
		return hash((self.block_id, self.mb_id, self.rank, self.op))


def complementary(op_type):
	"""
	The complementary is the opposite communication, that has the same peer.
	"""
	return {
		OperationType.RECV_FORWARD: (OperationType.SEND_BACKWARD, OperationType.SEND_FORWARD),
		OperationType.RECV_BACKWARD: (OperationType.SEND_FORWARD, OperationType.SEND_BACKWARD),
		OperationType.SEND_FORWARD: (OperationType.RECV_BACKWARD, OperationType.RECV_FORWARD),
		OperationType.SEND_BACKWARD: (OperationType.RECV_FORWARD, OperationType.RECV_BACKWARD),
	}[op_type]


def matching(op_type):
	"""
	The matching operation is the opposite communication, in the same direction.
	For instance, a recv(forward) is matched with a send(forward), and a send(backward) with a recv(backward).
	"""
	return {
		OperationType.RECV_FORWARD: OperationType.SEND_FORWARD,
		OperationType.RECV_BACKWARD: OperationType.SEND_BACKWARD,
		OperationType.SEND_FORWARD: OperationType.RECV_FORWARD,
		OperationType.SEND_BACKWARD: OperationType.RECV_BACKWARD,
	}[op_type]


comm_types = {
	OperationType.RECV_FORWARD,
	OperationType.RECV_BACKWARD,
	OperationType.SEND_FORWARD,
	OperationType.SEND_BACKWARD,
}


def resolve_one_pair(ops1, ops2):
	"""
	Find the blocking pattern between a pair of ranks
	"""
	if len(ops1) == 0 or len(ops2) == 0:
		return
	rank1 = ops1[0].rank
	rank2 = ops2[0].rank
	# assert rank1 > rank2, f"rank1 should be greater than rank2, got {rank1} and {rank2}"

	def find_matching_op(op):
		"""
		Find the corresponding recv or send on the other rank
		"""
		for other in ops2:
			if op.mb_id == other.mb_id and op.op == matching(other.op) and get_peer(other) == op.block_id:
				return other
		return None

	assert len(ops1) == len(ops2), (
		f"ops1 and ops2 should have the same length, got {len(ops1)} and {len(ops2)}"
	)

	# Find consecutive complementary operations in ops1
	for i in range(len(ops1) - 1):
		op1 = ops1[i]
		op2 = ops1[i + 1]
		if op1.op not in comm_types or op2.op not in comm_types:
			continue

		# If it's already batched, there is no issue here ; we can skip
		if (
			op1.options.get(OpOptions.BATCHED_COMM, None) is not None
			or op2.options.get(OpOptions.BATCHED_COMM, None) is not None
		):
			continue

		# Check if operations are complementary
		if op2.op in complementary(op1.op):
			op3 = find_matching_op(op2)
			op4 = find_matching_op(op1)

			if ops2.index(op3) > ops2.index(op4):
				# Not blocking ! That's the regular (send/recv) (recv/send) pattern
				continue

			# If op1 and op2 are not batched, op3 and op4 should not be batched either
			assert (
				op3.options.get(OpOptions.BATCHED_COMM, None) is None
				and op4.options.get(OpOptions.BATCHED_COMM, None) is None
			), "Operations should not be batched already"

			id_ = (rank1, rank2, i + 1)  # Unique id for pair + batch
			if not dist.is_initialized() or dist.get_rank() == 0:
				logger.debug(
					f"Marked operations for batched communication: {op1}, {op2}, {op3}, {op4} with id {id_}"
				)
			op1.options[OpOptions.BATCHED_COMM] = id_
			op2.options[OpOptions.BATCHED_COMM] = id_
			op3.options[OpOptions.BATCHED_COMM] = id_
			op4.options[OpOptions.BATCHED_COMM] = id_


def get_peer(op):
	"""
	Get the peer block id of a communication
	"""
	return op.options.get("dst", op.options.get("src", None))


def get_peer_rank(op, placement):
	"""
	Get the peer rank of a communication
	"""
	peer = get_peer(op)
	if peer is None:
		return None
	return placement[peer]


def mark_batched_comms(schedule, placement):
	"""
	Finds all operations that need to be batched and marks them for execution
	"""

	ranks = []
	# Unique, reverse order
	for p in reversed(placement):
		if p not in ranks:
			ranks.append(p)

	visited = set()
	for i in ranks:
		for j in ranks:
			if i == j or j in visited:
				continue
			ops1 = [op for op in schedule if op.rank == i and get_peer_rank(op, placement) == j]
			ops2 = [op for op in schedule if op.rank == j and get_peer_rank(op, placement) == i]
			resolve_one_pair(ops1, ops2)
		visited.add(i)


def schedule_to_str(schedule, print_comms=False):
	reprs = {
		OperationType.RECV_FORWARD: "rf",
		OperationType.FORWARD: "f",
		OperationType.SEND_FORWARD: "sf",
		OperationType.RECV_BACKWARD: "rb",
		OperationType.BACKWARD_INPUTS: "b",
		OperationType.BACKWARD_PARAMS: "w",
		OperationType.SEND_BACKWARD: "sb",
		OperationType.RECOMPUTE_FORWARD: "R",
		OperationType.RECOMPUTE_BACKWARD_INPUTS: "R*",
		OperationType.ALL_REDUCE_PARAM_GRADS: "(AR)",
	}

	def shorten(op):
		letter = reprs[op.op]
		if op.op == OperationType.FORWARD:
			is_recomputed = not op.options.get(OpOptions.SAVE, True)
			is_recomputed |= bool(op.options.get(OpOptions.REMAT_STRATEGY, False))
			letter = "F" if is_recomputed else "f"
		elif op.op == OperationType.BACKWARD_INPUTS:
			match op.options.get(OpOptions.SAVE, "full"):
				case "full":
					letter = "b"
				case "gradients":
					letter = "b*"
				case "none":
					letter = "B"
		return letter

	comm_types = {
		OperationType.RECV_FORWARD,
		OperationType.RECV_BACKWARD,
		OperationType.SEND_FORWARD,
		OperationType.SEND_BACKWARD,
	}
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
				[f"{shorten(op)}{op.mb_id if op.mb_id is not None else ''}" for op in rank_ops],
			)
		)
		lines.append(f"Rank {rank}: {ops_str}")
	return "\n".join(lines)


def check_schedule_validity(schedule):
	n_ranks = len(set(op.rank for op in schedule))
	n_mb = len(set(op.mb_id for op in schedule if op.mb_id is not None))

	def find(nodes, cond):
		for i, node in enumerate(nodes):
			if cond(node):
				return i, node
		return -1, None

	for rank in range(n_ranks):
		rank_ops = [op for op in schedule if op.rank == rank]
		rank_mb_ids = set(op.mb_id for op in rank_ops if op.mb_id is not None)
		assert len(rank_mb_ids) == n_mb, (
			f"[Rank {rank}] has {len(rank_mb_ids)} microbatches, expected {n_mb}"
		)

		for mb_id in rank_mb_ids:
			mb_ops = [op for op in rank_ops if op.mb_id == mb_id]
			i, _ = find(mb_ops, lambda op: op.op == OperationType.FORWARD)
			# in case of no backward, we allow to have 0 forward without remat
			j, _ = find(mb_ops, lambda op: op.op == OperationType.BACKWARD_INPUTS)
			assert i != -1 or j == -1, f"[Rank {rank}] Forward should be present for microbatch {mb_id}"
			assert i < j or j == -1, (
				f"[Rank {rank}] Forward should be before backward inputs for microbatch {mb_id}"
			)  # we allow "no backward"
			k, _ = find(mb_ops, lambda op: op.op == OperationType.BACKWARD_PARAMS)
			assert k != -1 or j == -1, (
				f"[Rank {rank}] Backward params should be present for microbatch {mb_id} since backward inputs is present"
			)
			assert j < k, (
				f"[Rank {rank}] Backward params should be after backward inputs for microbatch {mb_id}"
			)
