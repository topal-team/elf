import os
import json
import time
import argparse
import pulp as pl

from stage_remat_multiwave import StageRematMultiwaveProblem
from rarltm_greedy import RARLTMGreedyProblem
from params import get_params_from_config
from base import RematSolution


def remove_useless_remat(strategy, sched):
	"""
	Deactivate remat when the result is used directly, e.g. forward remat when doing F B.
	"""
	for local_sched in sched:
		for i in range(len(local_sched) - 1):
			op, block_id, mb_id = local_sched[i]
			next_op, next_block_id, next_mb_id = local_sched[i + 1]

			if block_id != next_block_id or mb_id != next_mb_id:
				continue

			if op == "f" and next_op == "b":
				strategy["full_fwd"][block_id][mb_id] = 0
			elif op == "b" and next_op == "w":
				strategy["activations_bwd"][block_id][mb_id] = 0
				strategy["full_bwd"][block_id][mb_id] = 0


def parse_method(method, scheduler):
	"""Parse a method string into (remat_strategy, scheduler, n_waves).

	Supports two formats:
	  - Dot-separated: "F-StageRemat.zbh1.1w" -> ("F-StageRemat", "zbh1", 1)
	  - Legacy hyphen-separated: "F-StageRemat-zbh1-1W" -> ("F-StageRemat", "zbh1", 1)

	When scheduler is provided separately, method is treated as the remat strategy name.
	"""
	if scheduler is not None:
		return method, scheduler, 1

	# Dot-separated format: "remat.scheduler.Nw"
	if "." in method:
		remat, sched, waves_str = method.rsplit(".", 2)
		waves = int(waves_str.rstrip("wW"))
		return remat, sched, waves

	# Legacy hyphen-separated format: "Method-scheduler-NW"
	*method_parts, sched, nwaves = method.split("-")
	nwaves = int(nwaves[:-1])
	return "-".join(method_parts), sched, nwaves


def run_solving(
	config_path,
	method,
	scheduler=None,
	mem=None,
	processors=None,
	nmb=None,
	time_limit=600,
	gap_limit=0.01,
	threads=1,
	solver="CPLEX_PY",
	verbose=False,
	force_uniform=False,
):
	"""
	Solve the problem and return the solution for the given method and config.

	Args:
		config_path (str): Path to the configuration file.
		method (str): Method to run.
		mem (int, optional): Memory of the GPU, in MB (overrides config).
		processors (int, optional): Number of GPUs to use (overrides derived value).
		time_limit (int, optional): Time limit for the ILP solver, in seconds.
		gap_limit (float, optional): Gap limit for the ILP solver.
		scheduler (str, optional): Scheduler to use. If not specified, we assume that the method name ends in "-[scheduler name]"
	Returns:
		dict: The solution dictionary.
	"""
	with open(config_path, "r") as f:
		config = json.load(f)

	method, scheduler, nwaves = parse_method(method, scheduler)
	config["scheduler"] = scheduler
	if processors is not None:
		config["ngpus"] = processors
	if nmb is not None:
		config["nmb"] = nmb
	if mem is not None:
		config["memgpu"] = mem

	params = get_params_from_config(config)

	if verbose:
		print(f"\n\n-- Running method '{method}' for config '{config_path}' --\n\n")

	# Decide between ILP and baseline paths
	ilp_methods = {"FB-StageRemat": True, "F-StageRemat": False}
	greedy_methods = {"FB-Greedy": True, "F-Greedy": False}
	baseline_methods = {"NoRemat": [0, 0, 0], "Uni-F-Remat": [1, 0, 0], "Uni-FB-Remat": [1, 0, 1]}

	if method in ilp_methods:
		allow_bwd = ilp_methods[method]
		# is_multiwave = params["p"] < params["nstages"]
		problem = (
			# StageRematProblem(method, params, allow_backward=allow_bwd)
			# if not is_multiwave
			StageRematMultiwaveProblem(
				method, params, allow_backward=allow_bwd, force_uniform=force_uniform
			)
		)
		start_time = time.time()
		result = problem.solve(time_limit=time_limit, solver=solver, threads=threads, gapRel=gap_limit)
		end_time = time.time()

		if verbose:
			problem.pretty_print_result()
			print(f"Solving took {end_time - start_time:.2f}s")

		if result == pl.LpStatusInfeasible:
			solution_json = {"error": "Infeasible"}
		else:
			solution_json = problem.get_solution()
	elif method in baseline_methods:
		# Build constant rematerialization strategy
		strategies = ["full_fwd", "activations_bwd", "full_bwd"]
		vec = baseline_methods[method]
		strategy = {
			strategies[i]: [[vec[i] for _ in range(params["m"])] for _ in range(params["nstages"])]
			for i in range(len(strategies))
		}
		remove_useless_remat(strategy, params["sched"])
		remat_solution = RematSolution(
			params,
			objective=-1,
			peak_mems=[0 for _ in range(params["p"])],
			placement="default",
			remat_strategy=strategy,
			schedule="default",
		)
		solution_json = remat_solution.to_json()
	elif method in greedy_methods:
		try:
			problem = RARLTMGreedyProblem(method, params, allow_backward=greedy_methods[method])
			result = problem.solve()

			if verbose:
				problem.pretty_print_result()

			if result == pl.LpStatusInfeasible:
				solution_json = {"error": "Infeasible"}
			else:
				solution_json = problem.get_solution()
		except Exception as e:
			print(f"Error solving greedy problem: {e}")
			solution_json = {"error": str(e)}
	else:
		raise ValueError(
			f"Unknown method '{method}'. Supported: {list(ilp_methods.keys()) + list(baseline_methods.keys())}"
		)

	return solution_json


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True, help="Path to the original runfile")
	parser.add_argument(
		"--profiling", type=str, required=True, help="Path to the profiling statistics"
	)
	parser.add_argument(
		"--method",
		type=str,
		required=True,
		help=(
			"Method to run: ILP (FB-StageRemat, F-StageRemat) or baselines (NoRemat, Uni-F-Remat, Uni-FB-Remat)"
		),
	)
	parser.add_argument(
		"--mem", "-M", type=int, default=None, help="Memory of the GPU, in MB (overrides config)"
	)
	parser.add_argument(
		"--processors",
		"-p",
		type=int,
		default=None,
		help="Number of GPUs to use (overrides derived value)",
	)
	parser.add_argument(
		"--nmb", type=int, default=None, help="Number of micro-batches (overrides derived value)"
	)
	parser.add_argument(
		"--time-limit", "-t", type=int, default=600, help="Time limit for the ILP solver, in seconds"
	)
	parser.add_argument("--threads", "-T", type=int, default=1, help="Number of threads to use")
	parser.add_argument("--scheduler", "-s", type=str, default=None, help="Scheduler to use")
	parser.add_argument("--output", "-o", type=str, default=None, help="Output file")
	parser.add_argument("--solver", "-S", type=str, default="CPLEX_PY", help="Solver to use")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	method, scheduler, nwaves = parse_method(args.method, args.scheduler)
	with open(args.config, "r") as f:
		config = json.load(f)
		processors = args.processors or config["ngpus"]
		mem = args.mem or config["memgpu"]
		nmb = args.nmb or config["nmb"]

	solution_json = run_solving(
		args.profiling,
		method,
		scheduler=scheduler,
		mem=mem,
		processors=processors,
		nmb=nmb,
		time_limit=args.time_limit,
		threads=args.threads,
		verbose=True,
		solver=args.solver,
	)

	if args.output:
		# Merge into existing JSON if present; otherwise start from the config JSON
		if os.path.exists(args.output):
			with open(args.output, "r") as f:
				data = json.load(f)
		else:
			try:
				with open(args.config, "r") as f:
					data = json.load(f)
			except Exception:
				data = {}
		# Attach solution under a dedicated key, preserving previous entries
		if "solutions" not in data:
			data["solutions"] = {}
		data["solutions"][args.method] = solution_json
		with open(args.output, "w") as f:
			json.dump(data, f, indent=1)
