import os

import pulp as pl
import argparse
import json

from base import LPProblem, RematSolution
from params import get_params_from_config


class StageRematMultiwaveProblem(LPProblem):
	def __init__(self, name, params, allow_backward=True, force_uniform=False):
		self.allow_backward = allow_backward
		self.force_uniform = force_uniform
		super().__init__(name, params)

	def init_variables(self):
		nstages = self.params["nstages"]
		m = self.params["m"]

		self.variables["fwd_remat"] = [
			[{} for _ in range(m)] for _ in range(nstages)
		]  # i, j, option name
		self.variables["bwd_remat"] = [
			[{} for _ in range(m)] for _ in range(nstages)
		]  # i, j, option name

		for i, stage in enumerate(self.params["stages"]):
			for j in range(m):
				for option in stage["fwd_remat_options"]:
					self.variables["fwd_remat"][i][j][option.name] = pl.LpVariable(
						f"{option.name}_{i}_{j}", cat=pl.LpBinary
					)
				for option in stage["bwd_remat_options"]:
					self.variables["bwd_remat"][i][j][option.name] = pl.LpVariable(
						f"{option.name}_{i}_{j}", cat=pl.LpBinary
					)

		super().init_variables()

	def formulate_constraints(self):
		super().formulate_constraints()
		m = self.params["m"]
		nstages = self.params["nstages"]
		for stage in range(nstages):
			for j in range(m):
				remat_variables = self.variables["fwd_remat"][stage][j]
				# Only one rematerialization option is used per micro-batch
				self.prob += pl.lpSum(remat_variables[name] for name in remat_variables) <= 1

				if self.force_uniform and j != 0:
					for option in remat_variables:
						self.prob += remat_variables[option] == self.variables["fwd_remat"][stage][0][option]

				remat_variables = self.variables["bwd_remat"][stage][j]
				if self.allow_backward:
					self.prob += pl.lpSum(remat_variables[name] for name in remat_variables) <= 1
					if self.force_uniform and j != 0:
						for option in remat_variables:
							self.prob += remat_variables[option] == self.variables["bwd_remat"][stage][0][option]

				else:
					for option in remat_variables:
						self.prob += remat_variables[option] == 0

	def remat_to_str(self, remat):
		return (
			"[" + ", ".join([str(i) + f"({remat[i]})" for i in range(len(remat)) if remat[i] != 0]) + "]"
		)

	def pretty_print_result(self):
		if super().pretty_print_result() == -1:
			return

		p = self.params["p"]
		m = self.params["m"]
		for i in range(p):
			pi = self.peak_mem_execution(i)
			ti = self.variables["T"][i].value()
			print(f"Processor {i}: {ti:.1f}ms - peak = {pi:.1f}MB")
			for stage in self.stages_ids[i]:
				for option in self.params["stages"][stage]["fwd_remat_options"]:
					print(
						f"\tStage {stage}: {option.name}: {self.remat_to_str([self.variables['fwd_remat'][stage][j][option.name].value() for j in range(m)])}"
					)
				for option in self.params["stages"][stage]["bwd_remat_options"]:
					print(
						f"\tStage {stage}: {option.name}: {self.remat_to_str([self.variables['bwd_remat'][stage][j][option.name].value() for j in range(m)])}"
					)

	def time_op(self, op, i, j):
		T = self.params["stages"][i]["T"]
		match op:
			case "f":
				t = T["f"]
			case "b":
				t = T["b"] + pl.lpSum(
					option.time_overhead * self.variables["fwd_remat"][i][j][option.name]
					for option in self.params["stages"][i]["fwd_remat_options"]
				)
			case "w":
				t = T["w"] + pl.lpSum(
					option.time_overhead * self.variables["bwd_remat"][i][j][option.name]
					for option in self.params["stages"][i]["bwd_remat_options"]
				)

		return t

	def mem_kept_op(self, op, i, j):
		M = self.params["stages"][i]["M"]

		match op:
			case "f":
				m = M["f"] - pl.lpSum(
					option.mem_freed * self.variables["fwd_remat"][i][j][option.name]
					for option in self.params["stages"][i]["fwd_remat_options"]
				)
			case "b":
				m = M["b"] - pl.lpSum(
					option.mem_freed * self.variables["bwd_remat"][i][j][option.name]
					for option in self.params["stages"][i]["bwd_remat_options"]
				)
			case "w":
				m = M["w"]

		return m

	def peak_mem_op(self, op, i, j):
		Mpeak = self.params["stages"][i]["Mpeak"]

		peakF = Mpeak["f"]
		peakB = Mpeak["b"]
		peakW = Mpeak["w"]
		peakNoGrad = Mpeak["f_no_grad"]
		mf = self.params["stages"][i]["M"]["f"]
		mb = self.params["stages"][i]["M"]["b"]
		if op == "f":
			# Peak of F is always the same, except when we full recompute, because then we can run under no_grad
			# peak = peakF + (self.variables["fwd_remat"][i][j]["full_fwd"] * (peakNoGrad - peakF))
			peak = peakF - pl.lpSum(
				option.mem_freed * self.variables["fwd_remat"][i][j][option.name]
				for option in self.params["stages"][i]["fwd_remat_options"]
			)
		elif op == "b":
			# Peak of B is either B or F if there is recomputation
			recompute_peak = max(peakF, mf + peakB)
			peak = peakB + (
				pl.lpSum(
					self.variables["fwd_remat"][i][j][option.name]
					for option in self.params["stages"][i]["fwd_remat_options"]
				)
			) * (recompute_peak - peakB)
			# peak = peakB
		elif op == "w":
			recompute_peak = max(peakF, mf + peakB, mb + peakW)
			peak = (
				peakW
				+ (self.variables["bwd_remat"][i][j]["full_bwd"] * (recompute_peak - peakW))
				+ (self.variables["bwd_remat"][i][j]["activations_bwd"] * (peakNoGrad - peakW))
			)
			# peak = peakW

		return peak

	def mparams(self, i):
		Mparams = sum(self.params["stages"][stage]["Mparams"] for stage in self.stages_ids[i])
		Mparams *= 2  # Params + Gradients, ignoring optimizer states
		return Mparams

	def get_solution(self):
		strategy = {}
		for i, stage in enumerate(self.params["stages"]):
			for option in stage["fwd_remat_options"]:
				if option.name not in strategy:
					strategy[option.name] = [
						[0 for _ in range(self.params["m"])] for _ in range(self.params["nstages"])
					]

				strategy[option.name][i] = [
					self.variables["fwd_remat"][i][j][option.name].value() for j in range(self.params["m"])
				]
			for option in stage["bwd_remat_options"]:
				if option.name not in strategy:
					strategy[option.name] = [
						[0 for _ in range(self.params["m"])] for _ in range(self.params["nstages"])
					]

				strategy[option.name][i] = [
					self.variables["bwd_remat"][i][j][option.name].value() for j in range(self.params["m"])
				]

		objective = self.prob.objective.value()
		peak_mems = [self.peak_mem_execution(i) for i in range(self.params["p"])]
		solution = RematSolution(
			self.params, objective=objective, peak_mems=peak_mems, remat_strategy=strategy
		)
		return solution.to_json()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Solve StageRematMultiwave problem")
	parser.add_argument("--config", type=str, required=True, help="Configuration file name")
	parser.add_argument("--profiled", "-p", type=str, required=True, help="Profiled data file name")
	parser.add_argument("--memgpu", "-M", type=int, default=None, help="Memory of the GPU, in MB")
	parser.add_argument("--scheduler", "-s", type=str, default=None, help="Scheduler to use")
	parser.add_argument(
		"--time-limit", "-t", type=int, default=60, help="Time limit for the ILP solver, in seconds"
	)
	parser.add_argument(
		"--allow_backward", "-ab", type=bool, default=True, help="Allow backward activations"
	)
	parser.add_argument(
		"--force_uniform", "-fu", action="store_true", help="Force uniform rematerialization"
	)
	parser.add_argument(
		"--output",
		"-o",
		type=str,
		default=None,
		help="Output file. The entry 'StageRematMultiwave' will be added to the file.",
	)
	args = parser.parse_args()

	with open(args.config, "r") as f:
		config = json.load(f)

	with open(args.profiled, "r") as f:
		profiled = json.load(f)

	config["stages"] = profiled["stages"]

	if args.memgpu:
		config["memgpu"] = args.memgpu
	if args.scheduler:
		config["scheduler"] = args.scheduler

	params = get_params_from_config(config)
	problem = StageRematMultiwaveProblem(
		"StageRematMultiwave",
		params,
		allow_backward=args.allow_backward,
		force_uniform=args.force_uniform,
	)
	problem.solve(time_limit=args.time_limit, solver="GUROBI")

	problem.pretty_print_result()
	if args.output:
		if os.path.exists(args.output):
			with open(args.output, "r+") as f:
				data = json.load(f)
		else:
			data = {}
		data["solutions"] = data.get("solutions", {})
		data["solutions"]["StageRematMultiwave"] = problem.get_solution()
		with open(args.output, "w") as f:
			json.dump(data, f)
