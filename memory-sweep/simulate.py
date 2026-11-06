"""
Solve the ILP for decreasing memory budgets and store the objective (estimated iteration time).

For every number of GPUs and every method, the memory budget starts at --start-mem and decreases by
2 * min(Mfp) until the problem becomes infeasible. Consecutive points without any rematerialization
are collapsed into the lowest one.

Output layout: <output-dir>/ngpus<p>/<method>.json, mapping memory budget (MB) -> solution.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline-ilps"))

from runall import run_solving  # noqa: E402

REMAT_TYPES = ["full_fwd", "activations_bwd", "full_bwd"]


def parse_args():
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument(
		"--profiled-dir", required=True, help="Directory with <prefix>-<n>stages.json profilings"
	)
	parser.add_argument("--prefix", default="ppcm21-s4096-h100", help="Profiling file prefix")
	parser.add_argument("--output-dir", required=True)
	parser.add_argument("--ngpus", type=int, nargs="+", default=[4, 8, 16])
	parser.add_argument(
		"--methods",
		nargs="+",
		default=[
			"F-StageRemat-1f1b-1W",
			"F-StageRemat-megatron-2W",
			"FB-StageRemat-zbv-2W",
			"FB-StageRemat-zbh2-1W",
		],
	)
	parser.add_argument(
		"--uniform", action="store_true", help="Force uniform remat across micro-batches"
	)
	parser.add_argument("--start-mem", type=int, default=100000, help="First memory budget (MB)")
	parser.add_argument(
		"--mems", type=int, nargs="+", default=None, help="Only solve for these memory budgets (MB)"
	)
	parser.add_argument("--time-limit", type=int, default=128)
	parser.add_argument("--gap-limit", type=float, default=0.01)
	parser.add_argument("--solver", default="CPLEX_PY")
	parser.add_argument("--threads", type=int, default=1)
	return parser.parse_args()


def is_no_remat(solution):
	return all(
		int(mb) == 0
		for rtype in solution
		if rtype in REMAT_TYPES or "remat" in rtype
		for rank in range(len(solution[rtype]))
		for mb in solution[rtype][rank]
	)


def solve(args, file, method, ngpus, mem):
	print(f"{ngpus} GPUs, {method}, M = {mem}...", end="\t", flush=True)
	start = time.time()
	solution = run_solving(
		file,
		method,
		mem=mem,
		processors=ngpus,
		time_limit=args.time_limit,
		threads=args.threads,
		solver=args.solver,
		gap_limit=args.gap_limit,
		force_uniform=args.uniform,
	)
	if "error" in solution:
		print(f"\033[91m{solution['error']}\033[0m")
	else:
		print(
			f"\033[92m{solution['objective'] / 1e3:.3f}s\033[0m\t(solving took {time.time() - start:.2f}s)"
		)
	return solution


def sweep(args, file, method, ngpus, step):
	data = {}
	mem = args.start_mem
	while mem > 0:
		solution = solve(args, file, method, ngpus, mem)
		if "error" in solution:
			break
		if is_no_remat(solution) and mem + step in data:
			del data[mem + step]
		data[mem] = solution
		mem -= step
	return data


def main():
	args = parse_args()
	for ngpus in args.ngpus:
		output_dir = os.path.join(args.output_dir, f"ngpus{ngpus}")
		os.makedirs(output_dir, exist_ok=True)
		for method in args.methods:
			nwaves = int(method.split("-")[-1][:-1])
			file = os.path.join(args.profiled_dir, f"{args.prefix}-{ngpus * nwaves}stages.json")
			with open(file, "r") as f:
				profiled = json.load(f)
			step = int(2 * min(stage["Mfp"] for stage in profiled["stages"] if stage["Mfp"] != 0))

			if args.mems:
				data = {mem: solve(args, file, method, ngpus, mem) for mem in args.mems}
				data = {mem: sol for mem, sol in data.items() if "error" not in sol}
			else:
				data = sweep(args, file, method, ngpus, step)

			with open(os.path.join(output_dir, f"{method}.json"), "w") as f:
				json.dump(data, f, indent=1)


if __name__ == "__main__":
	main()
