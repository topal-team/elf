from elf.registry import SCHEDULERS, resolve
from elf.scheduling import OperationType
from elf.topology import Placement


def get_balance(params, placement):
	"""
	Compute the balance of the pipeline
	"""
	nstages = len(placement)
	n = len(params["stages"])
	return [n // nstages + (1 if i < n % nstages else 0) for i in range(nstages)]


def translate_op(op):
	"""
	Translate the Operation object from the runtime to a simplified format used by the ILPs
	"""
	opchar = ""
	match op.op:
		case OperationType.FORWARD:
			opchar = "f"
		case OperationType.BACKWARD_INPUTS:
			opchar = "b"
		case OperationType.BACKWARD_PARAMS:
			opchar = "w"
		case _:
			return None

	return opchar, op.block_id, op.mb_id


def generate_schedule(placement, m, scheduler):
	p = len(set(placement))
	scheduler = resolve(scheduler, SCHEDULERS)
	schedule = scheduler(Placement(placement), m)
	simplified_schedule = [[] for _ in range(p)]
	for op in schedule:
		simplified_op = translate_op(op)
		if simplified_op:
			simplified_schedule[op.rank].append(simplified_op)

	return simplified_schedule
