import pulp as pl

from stage_remat_multiwave import StageRematMultiwaveProblem


class RARLTMGreedyProblem(StageRematMultiwaveProblem):
	def __init__(self, name, params, allow_backward=False):
		super().__init__(name, params, allow_backward=allow_backward)

	def init_variables(self):
		super().init_variables()
		nstages = self.params["nstages"]

		for i in range(nstages):
			for j in range(self.params["m"]):
				for option in self.params["stages"][i]["fwd_remat_options"]:
					self.variables["fwd_remat"][i][j][option.name].setInitialValue(0)
				for option in self.params["stages"][i]["bwd_remat_options"]:
					self.variables["bwd_remat"][i][j][option.name].setInitialValue(0)

	def formulate_constraints(self):
		super().formulate_constraints()

	def solve(self, *args, **kwargs):
		def increase_remat_level(op, stage, j):
			RF = self.variables["fwd_remat"][stage][j]["full_fwd"]
			RBF = self.variables["bwd_remat"][stage][j]["activations_bwd"]
			RBB = self.variables["bwd_remat"][stage][j]["full_bwd"]
			if op == "f":
				if RF.value() == 0:
					RF.setInitialValue(1)
				else:
					raise ValueError("Full forward remat is already used")
			elif op == "b":
				if RBF.value() == 0:
					RBF.setInitialValue(1)
				elif RBB.value() == 0:
					RBF.setInitialValue(0)
					RBB.setInitialValue(1)
				else:
					raise ValueError("Full backward remat is already used")

		p = self.params["p"]
		mgpu = self.params["MGpu"]
		for i in range(p):
			allowed_gap = (
				max(peak for peak in self.params["stages"][i]["Mpeak"].values()) * 1.0
			)  # margin if needed
			acc = self.mparams(i)
			peak = 0
			sched = self.params["sched"][i]
			for k in range(len(sched)):
				op, stage, j = sched[k]
				peak = max(acc + pl.value(self.peak_mem_op(op, stage, j)), peak)
				delta = pl.value(self.delta_mem_op(op, stage, j))

				if op == "f" or (op == "b" and self.allow_backward):
					while acc + delta > (mgpu - allowed_gap):
						# print(f"OP would not fit: current acc={acc:.1f}, delta={delta:.1f}, allowed_gap={allowed_gap:.1f}\nIncreasing remat level of {op} at stage {stage} and micro-batch {j}")
						increase_remat_level(op, stage, j)
						delta = pl.value(self.delta_mem_op(op, stage, j))

				acc += pl.value(delta)

		# Freeze variables
		nstages = self.params["nstages"]
		for i in range(nstages):
			for j in range(self.params["m"]):
				for option in self.params["stages"][i]["fwd_remat_options"]:
					self.variables["fwd_remat"][i][j][option.name].fixValue()
				for option in self.params["stages"][i]["bwd_remat_options"]:
					self.variables["bwd_remat"][i][j][option.name].fixValue()
		# Solve
		return super().solve(*args, **kwargs)
