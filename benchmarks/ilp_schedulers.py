import sys

sys.path.append(".")
from elf.scheduling import OpOptions, OperationType, Operation
from elf.scheduling.schedulers import _add_forward_pass, _add_backward_pass, _add_backward_params
from models.simple import Attention, TransformerBlock


class RematScheduler:
	"""Scheduler for unified rematerialization strategy solutions"""

	def __init__(self, solution):
		self.solution = solution

	def __call__(self, placement, nmb, signatures):
		# Create a new scheduler on the fly, that uses the order given from the solution
		order = self.solution["order"]

		def new_base_scheduler(placement, nmb, signatures):
			schedule = []
			for block_id, optype, mb_id in order:
				match optype:
					case "f":
						_add_forward_pass(
							schedule, placement, block_id, mb_id, placement[block_id], signatures[block_id]
						)
					case "b":
						_add_backward_pass(
							schedule, placement, block_id, mb_id, placement[block_id], signatures[block_id]
						)
					case "w":
						_add_backward_params(schedule, block_id, mb_id, placement[block_id])

			return schedule

		schedule = new_base_scheduler(placement, nmb, signatures)
		for op in schedule.copy():
			if op.op not in [OperationType.FORWARD, OperationType.BACKWARD_INPUTS]:
				continue

			self._handle_forward_remat(op)
			self._handle_backward_remat(op, schedule)

		return schedule

	def _find_operation(self, sched, optype, block_id, mb_id):
		for i, op in enumerate(sched):
			if op.mb_id == mb_id and op.op == optype and op.block_id == block_id:
				return i

		return None

	def _insert_recompute_operation(self, sched, block_id, mb_id, rank, op_type, **options):
		"""Helper to insert recomputation operations"""
		remat_op = Operation(block_id, mb_id, op_type, rank, **options)
		w_idx = self._find_operation(sched, OperationType.BACKWARD_PARAMS, block_id, mb_id)
		if w_idx is not None:
			sched.insert(w_idx, remat_op)
			return remat_op

		return None

	def _handle_forward_remat(self, op):
		if op.op != OperationType.FORWARD:
			return

		recompute_all_activations = self.solution["full_fwd"][op.block_id][op.mb_id]
		recompute_selective_activations = self.solution["selective_fwd"][op.block_id][op.mb_id]

		def checkpoint_strategy(
			name, module, rf=recompute_all_activations, rfsr=recompute_selective_activations
		):
			n = name.split(".")[0]
			try:
				n = int(n)
			except ValueError:
				return False

			if n < rf:
				return isinstance(module, TransformerBlock)
			if n >= rf and n < rf + rfsr:
				return isinstance(module, Attention)

			# No recomputation
			return False

		op.options[OpOptions.REMAT_STRATEGY] = checkpoint_strategy

	def _handle_backward_remat(self, op, sched):
		recompute_only_activations = self.solution["activations_bwd"][op.block_id][op.mb_id]
		recompute_all = self.solution["full_bwd"][op.block_id][op.mb_id]

		if op.op != OperationType.BACKWARD_INPUTS or (
			recompute_only_activations == 0 and recompute_all == 0
		):
			return

		# Recompute activations strategy
		def rbf_strategy(name, module, rbf=recompute_only_activations + recompute_all):
			n = name.split(".")[0]
			try:
				n = int(n)
			except ValueError:
				return False

			if n < rbf:
				return isinstance(module, TransformerBlock)
			return False

		op.options[OpOptions.RBF_STRATEGY] = rbf_strategy

		# Insert recompute activations operation
		remat_fwd = self._insert_recompute_operation(
			sched,
			op.block_id,
			op.mb_id,
			op.rank,
			OperationType.RECOMPUTE_FORWARD,
			**{OpOptions.RBF_STRATEGY: rbf_strategy},
		)

		if recompute_all == 0:
			return

		# Recompute gradients
		if remat_fwd:
			remat_fwd.options[OpOptions.SAVE] = True

			# RBB strategy
			def rbb_strategy(name, module, rbb=recompute_all):
				n = name.split(".")[0]
				try:
					n = int(n)
				except ValueError:
					return False, False

				is_recomputed = isinstance(module, TransformerBlock) and n < rbb
				is_frontier = isinstance(module, TransformerBlock) and n == rbb - 1
				return is_recomputed, is_frontier

			# Apply RBB strategy to all relevant operations
			op.options[OpOptions.RBB_STRATEGY] = rbb_strategy
			remat_fwd.options[OpOptions.RBB_STRATEGY] = rbb_strategy

			fwd_idx = self._find_operation(sched, OperationType.FORWARD, op.block_id, op.mb_id)
			if fwd_idx is not None:
				sched[fwd_idx].options[OpOptions.RBB_STRATEGY] = rbb_strategy

			# Insert recompute backward operation
			self._insert_recompute_operation(
				sched, op.block_id, op.mb_id, op.rank, OperationType.RECOMPUTE_BACKWARD_INPUTS
			)
