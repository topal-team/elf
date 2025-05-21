import sys

sys.path.append(".")
from elf.pipeline import Pipeline
from elf.scheduling import OpOptions, OperationType, Operation
from models.simple import Attention, TransformerBlock


class SchedulerBase:
	def __init__(self, base_scheduler):
		self.base_scheduler = Pipeline._get_scheduler(base_scheduler)

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


class FullRematScheduler(SchedulerBase):
	"""Scheduler that performs full recomputation of blocks"""

	def __init__(self, mbs, base_scheduler, factors):
		super().__init__(base_scheduler)
		self.mbs = self.round_to_int(mbs)
		self.factors = factors

	def round_to_int(self, solution):
		for key in solution:
			for i in range(len(solution[key])):
				solution[key][i] = [round(x) for x in solution[key][i]]

		return solution

	def __call__(self, placement, nmb, signatures):
		sched = self.base_scheduler(placement, nmb, signatures)

		for op in sched.copy():
			if op.op not in [OperationType.FORWARD, OperationType.BACKWARD_INPUTS]:
				continue

			self._handle_forward_remat(op)
			self._handle_backward_remat(op, sched)

		return sched

	def _handle_forward_remat(self, op):
		"""Handle forward pass recomputation settings"""
		if self.mbs["rf"][op.block_id][op.mb_id] == 1:
			op.options[OpOptions.REMAT_STRATEGY] = lambda name, _: name == ""

		elif self.mbs["rfsr"][op.block_id][op.mb_id] == 1:
			op.options[OpOptions.REMAT_STRATEGY] = lambda _, module: isinstance(module, Attention)

	def _handle_backward_remat(self, op, sched):
		"""Handle backward pass recomputation settings"""
		if not self.mbs["rbf"][op.block_id][op.mb_id]:
			return

		if op.op != OperationType.BACKWARD_INPUTS:
			return

		def strategy(_, module):
			return isinstance(module, TransformerBlock)

		op.options[OpOptions.RBF_STRATEGY] = strategy

		remat_op = self._insert_recompute_operation(
			sched,
			op.block_id,
			op.mb_id,
			op.rank,
			OperationType.RECOMPUTE_FORWARD,
			**{OpOptions.RBF_STRATEGY: strategy},
		)

		if self.mbs["rbb"][op.block_id][op.mb_id] == 1:
			self._handle_rbb_remat(op, remat_op, sched)

	def _handle_rbb_remat(self, op, remat_op, sched):
		"""Handle recompute backward blocks settings"""
		if op.op != OperationType.BACKWARD_INPUTS:
			return

		def rbb_strategy(name, module, block_id=op.block_id):
			n = name.split(".")[0]
			try:
				n = int(n)
			except ValueError:
				return False, False

			is_recomputed = isinstance(module, TransformerBlock)
			is_frontier = isinstance(module, TransformerBlock) and n == self.factors[block_id] - 1
			return is_recomputed, is_frontier

		op.options[OpOptions.RBB_STRATEGY] = rbb_strategy

		if remat_op:
			remat_op.options[OpOptions.RBB_STRATEGY] = rbb_strategy
			remat_op.options[OpOptions.SAVE] = True

			# Find and update forward operation
			fwd_idx = self._find_operation(sched, OperationType.FORWARD, op.block_id, op.mb_id)
			if fwd_idx is not None:
				sched[fwd_idx].options[OpOptions.RBB_STRATEGY] = rbb_strategy

		self._insert_recompute_operation(
			sched, op.block_id, op.mb_id, op.rank, OperationType.RECOMPUTE_BACKWARD_INPUTS
		)


class PartialRematScheduler(SchedulerBase):
	"""Scheduler that performs partial recomputation of blocks"""

	def __init__(self, mbs, base_scheduler):
		super().__init__(base_scheduler)
		self.mbs = self.round_to_int(mbs)

	def round_to_int(self, mbs):
		for key in mbs:
			if key == "b":
				mbs[key] = [round(x) for x in mbs[key]]
			else:
				mbs[key] = [[round(x) for x in mbs[key][i]] for i in range(len(mbs[key]))]

		return mbs

	def __call__(self, placement, nmb, signatures):
		sched = self.base_scheduler(placement, nmb, signatures)

		for op in sched.copy():
			if op.op not in [OperationType.FORWARD, OperationType.BACKWARD_INPUTS]:
				continue

			self._handle_forward_remat(op)
			self._handle_backward_remat(op, sched)

		return sched

	def _handle_forward_remat(self, op):
		if op.op != OperationType.FORWARD:
			return

		rf = self.mbs["rf"][op.block_id][op.mb_id]
		rfsr = self.mbs["rfsr"][op.block_id][op.mb_id]

		def checkpoint_strategy(name, module, rf=rf, rfsr=rfsr):
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
		rbf = self.mbs["rbf"][op.block_id][op.mb_id]
		rbb = self.mbs["rbb"][op.block_id][op.mb_id]

		if rbf == 0 or op.op != OperationType.BACKWARD_INPUTS:
			return

		# RBF strategy
		def rbf_strategy(name, module):
			n = name.split(".")[0]
			try:
				n = int(n)
			except ValueError:
				return False

			if n < rbf:
				return isinstance(module, TransformerBlock)
			return False

		op.options[OpOptions.RBF_STRATEGY] = rbf_strategy

		# Insert recompute forward operation
		remat_fwd = self._insert_recompute_operation(
			sched,
			op.block_id,
			op.mb_id,
			op.rank,
			OperationType.RECOMPUTE_FORWARD,
			**{OpOptions.RBF_STRATEGY: rbf_strategy},
		)

		if rbb != 0:
			assert rbb <= rbf, "rbb must be less than or equal to rbf"
			if remat_fwd:
				remat_fwd.options[OpOptions.SAVE] = True

				# RBB strategy
				def rbb_strategy(name, module):
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
