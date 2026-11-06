import argparse
import datetime
import json
import os
import time

import orloge as ol
import pandas as pd
import pulp as pl

from params import get_params_from_config
from stage_remat_multiwave import StageRematMultiwaveProblem
from rarltm_greedy import RARLTMGreedyProblem


def parse_args():
	parser = argparse.ArgumentParser(description="Run experiment")
	parser.add_argument(
		"--config-file", "-c", type=str, required=True, help="Path to the config file"
	)
	parser.add_argument(
		"--profiled-file", "-p", type=str, required=True, help="Path to the profiled file"
	)
	parser.add_argument("--scheduler", "-s", type=str, required=True, help="Scheduler to use")
	parser.add_argument(
		"--output",
		"-o",
		type=str,
		required=True,
		help="Path to save the results JSON file (default: results/solving_time.json)",
	)
	parser.add_argument(
		"--time-limit", "-t", type=int, default=3600, help="Time limit (in seconds) for a single solve"
	)
	parser.add_argument(
		"--name",
		"-n",
		type=str,
		default=None,
		help="Key to write the results in in the output JSON file",
	)
	parser.add_argument("--memgpu", "-M", type=int, default=None, help="Memory of the GPU, in MB")
	parser.add_argument("--warmstart", "-w", action="store_true", help="Warmstart the solver")
	parser.add_argument("--solver", "-S", type=str, default="PULP_CBC_CMD", help="Solver to use")
	return parser.parse_args()


def format_seconds(s):
	if s < 60:
		return f"{s}s"
	elif s < 3600:
		return f"{s // 60}min {s % 60}s"
	else:
		return f"{s // 3600}h {s % 3600 // 60}min {s % 60}s"


_ORLOGE_SOLVER_NAME = {
	"PULP_CBC_CMD": "CBC",
	"CPLEX_PY": "CPLEX",
	"CPLEX_CMD": "CPLEX",
	"GUROBI": "GUROBI",
}


def _warmstart_greedy(problem, params):
	greedy = RARLTMGreedyProblem(problem.name, params, allow_backward=problem.allow_backward)
	try:
		greedy.solve()
		print("Using greedy strategy to warmstart the problem.")
	except ValueError:
		print("Greedy strategy cannot find a working solution.")
		pass  # just use the incomplete solution

	problem.warmstart(greedy.get_variables_values(), fix=False)


def run_experiment():
	args = parse_args()
	with open(args.profiled_file, "r") as f:
		profiled = json.load(f)
	with open(args.config_file, "r") as f:
		config = json.load(f)
	config["stages"] = profiled["stages"]
	config["scheduler"] = args.scheduler or config["scheduler"]
	if args.memgpu:
		config["memgpu"] = args.memgpu
	params = get_params_from_config(config)

	if not os.path.exists(args.output):
		results = {}
	else:
		with open(args.output, "r") as f:
			results = json.load(f)

	for name, problem_type, allow_backward in [("FB-StageRemat", StageRematMultiwaveProblem, True)]:
		print(f"\n\n-- Running {name} --\n")

		problem = problem_type(name, params, allow_backward=allow_backward)

		if args.warmstart:
			_warmstart_greedy(problem, params)

		limit = args.time_limit
		now = time.time()
		end = now + limit
		end_str = datetime.datetime.fromtimestamp(end).strftime("%H:%M:%S")

		print(f" {limit}s ({format_seconds(limit)}) time limit (end: {end_str}): ", end="", flush=True)

		_ = problem.solve(args.solver, time_limit=limit, logPath=f"/tmp/{name}.log", threads=1)

		elapsed = time.time() - now

		solution_status = (
			problem.prob.sol_status
		)  # Problem status != solution status ! https://github.com/coin-or/pulp/discussions/382

		if solution_status == pl.LpStatusOptimal:
			print(f"  Solution found in {format_seconds(int(elapsed))}")
		elif solution_status == pl.LpStatusInfeasible:
			print(f"  Problem is infeasible in {format_seconds(int(elapsed))}")
		else:
			print(f"  No optimal solution found in {format_seconds(int(elapsed))}")

		data = ol.get_info_solver(f"/tmp/{name}.log", _ORLOGE_SOLVER_NAME[args.solver])
		final_gap = data.get("gap") if isinstance(data, dict) else None

		prog_raw = data.get("progress") if isinstance(data, dict) else None
		if prog_raw is not None:
			prog = pd.DataFrame(prog_raw)
			if not prog.empty:
				if "Gap" not in prog.columns:
					prog["BestInteger"] = pd.to_numeric(prog["BestInteger"])
					prog["CutsBestBound"] = pd.to_numeric(prog["CutsBestBound"])
					prog["Time"] = pd.to_numeric(prog["Time"])

					prog["Gap"] = (
						100 * abs(prog["BestInteger"] - prog["CutsBestBound"]) / (prog["CutsBestBound"] + 1e-10)
					)
				else:
					# Gap contains cells as a "XX%" string. Remove the "%" and convert to float
					prog["Gap"] = prog["Gap"].str.rstrip("%").astype(float)
				progress_records = prog.to_dict(orient="records")
			else:
				progress_records = []
		else:
			progress_records = []

		print("")
		print(f"  Final gap: {final_gap}")
		print(f"  Total solve time: {format_seconds(int(elapsed))}")

		name = args.name if args.name is not None else name
		results[name] = {
			"time_limit": limit,
			"elapsed": elapsed,
			"gap": final_gap,
			"solution_status": solution_status,
			"progress": progress_records,
		}

		with open(args.output, "w") as f:
			json.dump(results, f)


if __name__ == "__main__":
	run_experiment()
