import pulp as pl
import matplotlib.pyplot as plt


def op_name_to_index(op):
	return ["f", "b", "w"].index(op)


# Unified solution representation for all ILPs/heuristics
class RematSolution:
	def __init__(
		self,
		params,
		objective,
		peak_mems,
		placement="default",
		remat_strategy="default",
		schedule="default",
	):
		self.params = params
		self.objective = objective
		self.peak_mems = peak_mems
		self.schedule = schedule if schedule != "default" else self.get_default_schedule()
		self.remat_strategy = (
			remat_strategy if remat_strategy != "default" else self.get_default_remat_strategy()
		)

	def get_default_schedule(self):
		"""
		Default schedule: use the one in params and translate it to the right format
		"""
		return [op for local_schedule in self.params["sched"] for op in local_schedule]

	def get_default_remat_strategy(self):
		"""
		Default strategy: no rematerialization
		(0 for each option, for each processor, for each microbatch)
		[option][processor][microbatch]
		"""
		remat_strategy = {}
		for stage in self.params["stages"]:
			for option in stage["fwd_remat_options"] + stage["bwd_remat_options"]:
				remat_strategy[option.name] = [
					[0 for _ in range(self.params["m"])] for _ in range(self.params["p"])
				]
		return remat_strategy

	def to_json(self):
		solution = {
			"placement": self.params["placement"],
			"objective": self.objective,
			"peak_mems": self.peak_mems,
			"order": self.schedule,
		}

		for option, value in self.remat_strategy.items():
			solution[option] = value

		return solution


class LPProblem:
	def __init__(self, name, params):
		self.name = name
		self.prob = pl.LpProblem(name, pl.LpMinimize)
		self.variables = {}
		self.params = params
		self.stages_ids = {
			rank: [i for i, k in enumerate(self.params["placement"]) if k == rank]
			for rank in range(self.params["p"])
		}
		self.init_variables()
		self.formulate_objective()
		self.formulate_constraints()

	def init_variables(self):
		p = self.params["p"]
		m = self.params["m"]
		nstages = len(self.params["placement"])
		ops = ["f", "b", "w"]
		n_ops = len(ops)

		self.lower_bounds = [
			sum(
				sum(self.params["stages"][stage]["T"][op] for op in ops) * m
				for stage in self.stages_ids[rank]
			)
			for rank in range(p)
		]

		# Ending time of operation k for micro-batch j on processor i
		self.variables["E"] = [
			[
				[pl.LpVariable(f"E_{stage}_{j}_{k}", cat=pl.LpContinuous, lowBound=0) for k in range(n_ops)]
				for j in range(m)
			]
			for stage in range(nstages)
		]

		self.variables["T"] = [None for _ in range(p)]
		# Iteration time on processor i
		for rank in range(p):
			first_op, first_stage, first_mb = self.params["sched"][rank][0]
			last_op, last_stage, last_mb = self.params["sched"][rank][-1]

			self.variables["T"][rank] = self.variables["E"][last_stage][last_mb][
				op_name_to_index(last_op)
			] - (
				self.variables["E"][first_stage][first_mb][op_name_to_index(first_op)]
				- self.time_op(first_op, first_stage, first_mb)
			)

		self.variables["Tmax"] = pl.LpVariable(
			"Maxtime", cat=pl.LpContinuous, lowBound=max(self.lower_bounds)
		)

	def formulate_objective(self):
		self.prob += self.variables["Tmax"]

	def formulate_constraints(self):
		p = self.params["p"]
		m = self.params["m"]
		T = self.variables["T"]
		nstages = len(self.params["placement"])
		Tmax = self.variables["Tmax"]

		for i in range(p):
			# Tmax is the maximum T[i]
			self.prob += Tmax >= T[i]

		ops = ["f", "b", "w"]
		E = self.variables["E"]
		self.prob += E[0][0][0] - self.time_op("f", 0, 0) == 0  # first operation starts at time 0

		MGpu = self.params["MGpu"]
		placement = self.params["placement"]

		for i in range(p):
			n_ops = len(ops) * m * len(self.stages_ids[i])
			memory_kept = [self.mparams(i)]
			for k in range(n_ops):
				op, stage, j = self.params["sched"][i][k]
				time_op = self.time_op(op, stage, j)
				delta_mem_op = self.delta_mem_op(op, stage, j)
				peak_mem_op = self.peak_mem_op(op, stage, j)

				# Inter-processor dependencies
				if op == "f" and stage != 0:
					idx = op_name_to_index("f")
					tcomm = self.params["stages"][stage - 1]["Tcomm"] if placement[stage - 1] != i else 0
					self.prob += E[stage][j][idx] >= E[stage - 1][j][idx] + time_op + tcomm
				elif op == "b" and stage != nstages - 1:
					idx = op_name_to_index("b")
					tcomm = self.params["stages"][stage]["Tcomm"] if placement[stage + 1] != i else 0
					self.prob += E[stage][j][idx] >= E[stage + 1][j][idx] + time_op + tcomm

				# Intra-processor dependencies
				if k != 0:
					op_prev, stage_prev, j_prev = self.params["sched"][i][k - 1]
					self.prob += (
						E[stage][j][op_name_to_index(op)]
						>= E[stage_prev][j_prev][op_name_to_index(op_prev)] + time_op
					)

				self.prob += pl.lpSum(memory_kept) + peak_mem_op <= MGpu
				memory_kept.append(delta_mem_op)

	def solve(self, solver="PULP_CBC_CMD", time_limit=60, *args, **kwargs):
		solver = getattr(pl, solver)
		self.prob.solve(solver(msg=False, timeLimit=time_limit, *args, **kwargs))
		return self.prob.status

	def time_op(self, op, i, j):
		raise NotImplementedError("Subclasses must implement this method")

	def mem_kept_op(self, op, i, j):
		raise NotImplementedError("Subclasses must implement this method")

	def delta_mem_op(self, op, i, j):
		match op:
			case "f":
				return self.mem_kept_op("f", i, j)
			case "b":
				return self.mem_kept_op("b", i, j) - self.mem_kept_op("f", i, j)
			case "w":
				return self.mem_kept_op("w", i, j) - self.mem_kept_op("b", i, j)

	def peak_mem_op(self, op, i, j):
		raise NotImplementedError("Subclasses must implement this method")

	def mparams(self, i):
		"""
		Memory taken by parameters, gradients, optimizer states, etc for processor `i`.
		"""
		raise NotImplementedError("Subclasses must implement this method")

	def pretty_print_result(self):
		if self.prob.sol_status == pl.LpSolutionOptimal:
			print(f"Optimal solution found: {self.prob.objective.value():.1f}ms")
			return 0
		elif self.prob.status == pl.LpStatusInfeasible:
			print("Problem is infeasible")
			return -1
		else:
			print(f"Found suboptimal solution: {self.prob.objective.value():.1f}ms")
			return 0

	def peak_mem_execution(self, i):
		acc = self.mparams(i)
		peak = 0
		sched = self.params["sched"][i]
		for k in range(len(sched)):
			op, stage, j = sched[k]
			peak = max(acc + pl.value(self.peak_mem_op(op, stage, j)), peak)
			acc += pl.value(self.delta_mem_op(op, stage, j))

		return peak

	def get_solution(self):
		raise NotImplementedError("Subclasses must implement this method")

	def get_variables_values(self):
		def resolve(v):
			# Minimal pytree traversal
			if isinstance(v, (pl.LpVariable, pl.LpAffineExpression)):
				return v.value()
			elif isinstance(v, (tuple, list)):
				return type(v)(resolve(x) for x in v)
			elif isinstance(v, dict):
				return {key: resolve(value) for key, value in v.items()}

		return resolve(self.variables)

	def warmstart(self, values, fix=False, check=False):
		def is_computed(var):
			return var.name.startswith("E_") or var.name.startswith("T_") or var.name == "Maxtime"

		def set_values(var, val):
			if isinstance(var, pl.LpVariable) and val is not None:
				var.setInitialValue(val, check=check)
				if fix and not is_computed(var):
					var.fixValue()
			elif isinstance(
				var, pl.LpAffineExpression
			):  # mandatory case because LpAffineExpression is a Dict subclass :)
				return
			elif isinstance(var, (tuple, list)):
				for v, value in zip(var, val):
					set_values(v, value)
			elif isinstance(var, dict):
				for key in var:
					set_values(var[key], val[key])

		set_values(self.variables, values)

	def display_solution(self):
		# Extract start and end times for each operation
		operations = []
		for i in range(self.params["p"]):
			sched = self.params["sched"][i]
			for k in range(len(sched)):
				op, stage, j = sched[k]
				end_time = pl.value(self.variables["E"][stage][j][op_name_to_index(op)])
				start_time = end_time - pl.value(self.time_op(op, stage, j))
				if start_time is not None and end_time is not None:
					operations.append(
						{
							"processor": i,
							"microbatch": j,
							"op": op,
							"stage": stage,
							"start": start_time,
							"end": end_time,
						}
					)

		if not operations:
			print("No operations to display")
			return

		# Create the Gantt chart
		fig, ax = plt.subplots(figsize=(12, 6))

		# Define colors for different operation types
		colors = {"f": "tab:blue", "b": "tab:orange", "w": "tab:green"}

		# Plot each operation as a horizontal bar
		for op in operations:
			ax.barh(
				op["processor"],
				op["end"] - op["start"],
				left=op["start"],
				height=0.5,
				color=colors.get(op["op"], "gray"),
				alpha=0.8,
				edgecolor="black",
				label=f"{op['op']}_{op['stage']}_{op['microbatch']}",
			)

			# Add text label in the middle of each bar
			ax.text(
				(op["start"] + op["end"]) / 2,
				op["processor"],
				f"{op['op']}_{op['stage']}_{op['microbatch']}",
				ha="center",
				va="center",
				color="black",
				fontsize=8,
			)

		# Remove duplicate labels
		handles, labels = [], []
		for h, label in zip(*ax.get_legend_handles_labels()):
			if label[0] not in [lab[0] for lab in labels]:
				handles.append(h)
				labels.append(label[0])

		ax.legend(handles, ["F", "B", "W"], loc="upper right", title="Operation Type")

		# Set labels and title
		ax.set_xlabel("Time (ms)")
		ax.set_ylabel("Processor")
		ax.set_title("Execution Schedule Gantt Chart")

		# Set y-ticks to processor numbers
		ax.set_yticks(range(self.params["p"]))
		ax.set_yticklabels([f"P{i}" for i in range(self.params["p"])])

		# Add grid lines
		ax.grid(True, axis="x", linestyle="--", alpha=0.7)

		# Show the plot
		plt.tight_layout()
		plt.show()
