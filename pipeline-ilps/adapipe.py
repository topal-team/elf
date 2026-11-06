# Custom implementation of the optimization process in AdaPipe https://dl.acm.org/doi/10.1145/3620666.3651359

import json
import math

from functools import lru_cache
from dataclasses import dataclass

from argparse import ArgumentParser

import matplotlib.pyplot as plt


def _scale_all_mems(
	mems: list[float], minput: list[float], mparams: list[float], M: float
) -> list[float]:
	"""
	Scale all memory values by their GCD to reduce numerical precision issues.

	Args:
		mems: Memory costs for each computation unit
		minput: Input memory sizes for each computation unit
		mparams: Parameter memory sizes for each computation unit
		M: Memory limit

	Returns:
		Scaled memory limit
	"""

	# Collect all values
	all_values = []
	all_values.extend(mems)
	all_values.extend(minput)
	all_values.extend(mparams)
	all_values.append(M)

	int_values = [int(v) for v in all_values if v > 0]

	if not int_values:
		return M

	# Find GCD of all values
	gcd = int_values[0]
	for val in int_values[1:]:
		gcd = math.gcd(gcd, val)

	# Scale everything down by the GCD
	scaling = gcd
	print(f"Scaling memory values down by {scaling}")

	for i in range(len(mems)):
		mems[i] = mems[i] / scaling
	for i in range(len(minput)):
		minput[i] = minput[i] / scaling
	for i in range(len(mparams)):
		mparams[i] = mparams[i] / scaling

	M = M / scaling

	return mems, minput, mparams, M


def fb_times(
	N: int,
	M: int,
	p: int,
	s: int,
	tf: list[float],
	tb: list[float],
	mem: list[float],
	minput: list[float],
) -> tuple[float, float]:
	"""
	DP optimization problem 1 - gets fixed forward time and minimal backward time
	O(mN)

	N: number of computation units
	M: memory limit
	p: number of processors
	s: stage index
	tf: forward time for each computation unit
	tb: backward time for each computation unit
	mem: memory cost for each computation unit
	minput: input memory size for each computation unit
	"""
	if M < 0:
		return math.inf, math.inf

	@lru_cache(maxsize=None)
	def tsim(s, i, m):
		if m < 1:
			return -math.inf  # forbid embedding (first layer)
		# Maximum saved recomputation cost with the first i units under the memory limit m for stage s
		if i == -1:
			return 0

		if i == N - 1:
			return tsim(s, i - 1, m - (p - s) * mem[i]) + tf[i]  # forbid head (last layer)

		return max(
			tsim(s, i - 1, m - ((p - s) * minput[i])), tsim(s, i - 1, m - ((p - s) * mem[i])) + tf[i]
		)

	assert len(tf) == len(tb) == len(mem) == len(minput) == N, (
		f"Length of tf, tb, mem, and minput must be equal to N: got {len(tf)}, {len(tb)}, {len(mem)}, {len(minput)} != {N}"
	)
	fgs = sum(tf)
	bgs = sum(tf) + sum(tb) - tsim(s, N - 1, M)
	return fgs, bgs


def Ws(s, p, F, B):
	if s == p - 1:
		return F[s]
	return max(Ws(s + 1, p, F, B) + B[s + 1], (p - s - 1) * F[s]) + F[s]


def Es(s, p, F, B):
	if s == p - 1:
		return B[s]
	return max(Es(s + 1, p, F, B) + F[s + 1], (p - s - 1) * B[s]) + B[s]


def Ms(s, p, F, B):
	if s == p - 1:
		return F[s] + B[s]
	return max(Ms(s + 1, p, F, B), F[s] + B[s])


def Ss(s, n, p, tf, tb):
	return (n - p + s) * Ms(s, p, tf, tb)


def partitioning(p, n, L, f, b):
	@dataclass
	class State:
		W: float = 0
		E: float = 0
		M: float = math.inf
		F: float = 0
		B: float = 0
		T: float = math.inf
		j: int = 0

	P = {
		(s, i): State() for s in range(p + 1) for i in range(L + 1)
	}  # P[s,i] = best partitioning strategy for the layer sequence from layer i to the last layer with stages from s to the last
	P[(p, L)].T = 0  # end of recursion
	P[(p, L)].M = 0
	for s in range(p - 1, -1, -1):
		for i in range(L - p + s + 1, -1, -1):
			for j in range(i, L - p + s + 1):
				# Force the last stage to end with the last layer
				if s == p - 1 and j != L - 1:
					continue
				# Same for the first stage to start with the first layer
				if s == 0 and i != 0:
					continue

				W = f[(s, i, j)] + max(P[s + 1, j + 1].W + P[s + 1, j + 1].B, (p - s - 1) * f[(s, i, j)])
				E = b[(s, i, j)] + max(P[s + 1, j + 1].E + P[s + 1, j + 1].F, (p - s - 1) * b[(s, i, j)])
				M = max(P[s + 1, j + 1].M, f[(s, i, j)] + b[(s, i, j)])
				F = f[(s, i, j)]
				B = b[(s, i, j)]
				T = W + E + (n - p + s) * M

				if T < P[s, i].T:
					P[s, i] = State(W, E, M, F, B, T, j)

	return P


def _peak_F(i: int, j: int, units: list[dict], recompute: list[int]) -> float:
	"""
	Get the peak memory of stage from unit i to j

	i: the start unit index
	j: the end unit index
	units: the list of units
	"""
	return max(
		[
			sum(
				units[prev]["M"][0] if prev not in recompute else units[prev]["Minput"]
				for prev in range(i, k)
			)
			+ units[k]["Mpeak"][0]
			for k in range(i, j + 1)
		]
		+ [0]
	)


def _peak_B(i: int, j: int, units: list[dict]) -> float:
	"""
	Get the peak memory of stage from unit i to j
	"""
	return units[j]["Mpeak"][1]


def _mem_F(i: int, j: int, units: list[dict], recompute: list[int]) -> float:
	"""
	Get the memory of stage from unit i to j
	"""
	return sum(units[k]["M"][0] for k in range(i, j + 1) if k not in recompute)


def solve_adapipe(opt_args: dict) -> dict:
	"""
	Solves the AdaPipe optimization problem for the given optimization arguments.
	Returns a list of tuples, each containing the stage index, the start layer index, the end layer index (both inclusive), and the recomputation strategy for the stage.
	"""
	L = opt_args["L"]
	p = opt_args["p"]
	n = opt_args["n"]
	M = opt_args["M"]
	tf = opt_args["tf"]
	tb = opt_args["tb"]
	mem = opt_args["mem"]
	minput = opt_args["minput"]
	Mparams = opt_args["Mparams"]

	f, b = {}, {}
	cache = {}
	for s in range(p):
		for i in range(0, L):
			for j in range(i, L):
				length = j - i
				parity = i % 2
				is_start = i == 0
				is_end = j == L - 1

				cache_key = (s, length, parity) if not (is_start or is_end) else None

				mem_limit = M - sum(Mparams[i : j + 1]) * 2  # params + gradients

				if cache_key is not None and cache_key in cache:
					f[s, i, j], b[s, i, j] = cache[cache_key]
				else:
					mem_limit = M - sum(Mparams[i : j + 1]) * 2  # params + gradients
					f[s, i, j], b[s, i, j] = fb_times(
						j - i + 1,
						mem_limit,
						p,
						s,
						tf[i : j + 1],
						tb[i : j + 1],
						mem[i : j + 1],
						minput[i : j + 1],
					)
					if cache_key is not None:
						cache[cache_key] = (f[s, i, j], b[s, i, j])

	P = partitioning(p, n, L, f, b)
	partition = []
	i = 0
	s = 0
	while s < p:
		j = P[s, i].j
		recompute = backtrack_recomputation(i, j, P[s, i].B, tf, tb)
		partition.append((s, i, j, recompute))
		i = j + 1
		s += 1

	return partition


def backtrack_recomputation(i, j, B, tf, tb):
	"""
	Backtracks the recomputation strategy for the given stage, start layer, and end layer.
	Returns a list of layer indices that should be recomputed.

	i: the start layer index
	j: the end layer index
	B: the backward time given by the DP optimization problem
	tf: the forward times of the layers in the stage
	tb: the backward times of the layers in the stage
	"""
	no_recomp = sum(tf[i : j + 1]) + sum(tb[i : j + 1])
	diff = no_recomp - B  # saved recomputations time = time if we recomputed everything - actual time

	# Find the subset of layers in [i, j+1] whose forward times sum to diff
	# This represents the layers that should be recomputed
	# Using dynamic programming to find exact subset sum solution

	n_layers = j - i + 1
	target = diff

	# DP table: dp[k][s] = True if we can achieve sum s using first k layers
	# We'll use a dictionary to avoid storing all possible sums
	dp = [{} for _ in range(n_layers + 1)]
	dp[0][0.0] = []  # Base case: 0 layers, 0 sum, no layers selected

	# Build DP table
	for k in range(1, n_layers + 1):
		layer_idx = i + k - 1
		time = tf[layer_idx]

		# Copy previous states (not including current layer)
		for s, layers in dp[k - 1].items():
			if s not in dp[k]:
				dp[k][s] = layers

		# Add states that include current layer
		for s, layers in dp[k - 1].items():
			new_sum = s + time
			if new_sum not in dp[k]:
				dp[k][new_sum] = layers + [layer_idx]

	# Find the closest sum to target in the final DP state
	best_sum = None
	best_diff = float("inf")
	for s in dp[n_layers]:
		if abs(s - target) < best_diff:
			best_diff = abs(s - target)
			best_sum = s

	layers_to_keep = dp[n_layers][best_sum] if best_sum is not None else []

	return [x for x in range(i, j + 1) if x not in layers_to_keep]


def parse_args():
	parser = ArgumentParser()
	parser.add_argument("--stats-file", type=str, default=None, help="Path to the stats file")
	parser.add_argument("--ngpus", "-p", type=int, default=None, help="Number of GPUs")
	parser.add_argument("--nmb", "-n", type=int, default=None, help="Number of micro-batches")
	parser.add_argument("--M", "-M", type=float, default=None, help="Memory limit")
	parser.add_argument("--tcomm", type=float, default=0.0, help="Communication time")
	parser.add_argument("--plot", action="store_true", help="Plot the partitioning")
	args = parser.parse_args()
	assert args.ngpus is not None or args.stats_file is None, (
		"If using a stats file, you must also provide the number of GPUs"
	)
	assert args.M is not None or args.stats_file is None, (
		"If using a stats file, you must also provide the memory limit"
	)
	return args


def sort_units(profiled_stats: dict) -> list[dict]:
	def find_by_name(name: str) -> dict:
		for unit in profiled_stats["units"]:
			if unit["name"] == name:
				return unit
		raise ValueError(f"Unit {name} not found")

	res = []

	res.append(find_by_name("embed"))
	for block in range(
		profiled_stats["model"]["n_blocks"]
	):  # this works as long as we keep the model architecture in the profiled stats json
		res.append(find_by_name(f"block_{block}_attention"))
		res.append(find_by_name(f"block_{block}_ffn"))
	res.append(find_by_name("head"))
	return res


def get_makespan(P: dict, tcomm: float) -> float:
	"""
	Get the makespan of the partitioning.
	"""

	p = max(P.keys())[0]
	return P[0, 0].T + tcomm * 2 * (p - 1)


def get_peak(P: dict, tf: list[float], tb: list[float], units: list[dict]) -> float:
	"""
	Get the peak memory of the partitioning.
	"""
	i = 0
	s = 0
	p = max(P.keys())[0]
	highest_peak = 0
	while s < p:
		j = P[s, i].j
		recompute = backtrack_recomputation(i, j, P[s, i].B, tf, tb)
		# Peak is at the end of warmup ; either during the last F (first case), or during the first B
		peak = (
			max(
				(p - s - 1) * _mem_F(i, j, units, recompute) + _peak_F(i, j, units, recompute),
				(p - s) * _mem_F(i, j, units, recompute) + _peak_B(i, j, units),
			)
			+ sum(units[k]["Mparams"] for k in range(i, j + 1)) * 2
		)
		highest_peak = max(peak, highest_peak)
		i = j + 1
		s += 1

	return highest_peak


def main():
	args = parse_args()
	if args.stats_file is not None:
		with open(args.stats_file, "r") as f:
			stats = json.load(f)
		L = len(stats["units"])
		p = args.ngpus
		L // p  # number of computation units per stage
		n = args.nmb or p * 2
		M = args.M  # memory limit
		units = sort_units(stats)
		tf = [unit["T"][0] for unit in units]
		tb = [unit["T"][1] for unit in units]
		mem = [unit["M"][0] for unit in units]
		minput = [unit["Minput"] for unit in units]
		mparams = [unit["Mparams"] for unit in units]
	else:
		L = 64  # number of layers
		p = 4  # number of processors
		L // p  # number of computation units per stage
		n = 8  # number of micro-batches
		M = 128  # memory limit
		tf = [1.5, 2] * (L // 2)
		tb = [3, 4] * (L // 2)
		mem = [8, 16] * (L // 2)
		minput = [1, 1] * (L // 2)
		mparams = [1, 1] * (L // 2)

	mem, minput, mparams, M = _scale_all_mems(mem, minput, mparams, M)

	# F, B = zip(
	# 	*[
	# 		fb_times(N, M, p, s, tf[s : s + N], tb[s : s + N], mem[s : s + N], minput[s : s + N])
	# 		for s in range(p)
	# 	]
	# )

	# print("Balanced partitioning: \t\t\t(Not accurate if L is not divisible by p)")
	# for s in range(p):
	# 	print(f"Stage {s}:", end="\t")
	# 	warmup = Ws(s, p, F, B)
	# 	steady = Ss(s, n, p, F, B)
	# 	end = Es(s, p, F, B)
	# 	total = warmup + steady + end
	# 	print(f"{warmup} - {steady} - {end} = {total} (Warmup / Steady / Cooldown)")

	print("-----------")

	f, b = {}, {}
	cache = {}  # Cache for isomorphic cases
	for s in range(p):
		for i in range(0, L):
			for j in range(i, L):
				# Check if this is an isomorphic case that can be cached
				# Cases are isomorphic if they have the same:
				# 1. Length (j - i)
				# 2. Parity of i
				# 3. Stage ID
				# But exclude cases that include start (i=0) or end (j=L-1) layers

				# (AdaPipe paper section 5.3)
				length = j - i
				parity = i % 2
				is_start = i == 0
				is_end = j == L - 1

				cache_key = (s, length, parity) if not (is_start or is_end) else None

				if cache_key is not None and cache_key in cache:
					# Use cached result
					f[s, i, j], b[s, i, j] = cache[cache_key]
				else:
					# Compute and potentially cache
					mem_limit = M - sum(mparams[i : j + 1]) * 2  # params + gradients
					f[s, i, j], b[s, i, j] = fb_times(
						j - i + 1,
						mem_limit,
						p,
						s,
						tf[i : j + 1],
						tb[i : j + 1],
						mem[i : j + 1],
						minput[i : j + 1],
					)

					# Cache the result if it's an isomorphic case
					if cache_key is not None:
						cache[cache_key] = (f[s, i, j], b[s, i, j])

	P = partitioning(p, n, L, f, b)
	i = 0
	s = 0
	while i < L and s < p:
		j = P[s, i].j
		print(
			f"Stage {s} from {i} to {j} ({j - i + 1} layers): {P[s, i].T} ms ({f[s, i, j]} + {b[s, i, j]} per mb)"
		)  # best partitioning for all layers starting from stage 0
		recompute = backtrack_recomputation(i, j, P[s, i].B, tf, tb)
		# print(
		# 	f"Peaks/Kept: {_peak_F(i, j, units, recompute)}/{_mem_F(i, j, units, recompute)} - {_peak_B(i, j, units)}/0"
		# )
		print("Recomputation: ", recompute)
		i = j + 1
		s += 1

	print(f"Makespan: {get_makespan(P, args.tcomm)}")
	print(f"Peak: {get_peak(P, tf, tb, units)} MB")

	# for s, i, j in [(0, 0, 14), (1, 15, 30), (2, 31, 46), (3, 47, 61)]:
	# 	print(f"---- {s} {i} {j} ----")

	# 	W = f[(s, i, j)] + max(P[s + 1, j + 1].W + P[s + 1, j + 1].B, (p - s) * f[(s, i, j)])
	# 	E = b[(s, i, j)] + max(P[s + 1, j + 1].E + P[s + 1, j + 1].F, (p - s) * b[(s, i, j)])
	# 	M = max(P[s + 1, j + 1].M, f[(s, i, j)] + b[(s, i, j)])
	# 	T = W + E + (n - p + s) * M

	# 	print("f", f[s, i, j])
	# 	print("b", b[s, i, j])
	# 	print("W", W)
	# 	print("E", E)
	# 	print("M", M)
	# 	print("T", T)

	if args.plot:
		plot_repartition(P, tf, tb)


def plot_repartition(P: dict, tf: list[float], tb: list[float]):
	"""
	Plot the number of layers per processor. Remove the start and end layer.
	"""
	L = max(P.keys())[1]
	p = max(P.keys())[0]
	i = 0
	s = 0
	layers = []
	recomp = []
	while i < L and s < p:
		j = P[s, i].j
		n = j - i + 1
		nrecomp = len(
			[r for r in backtrack_recomputation(i, j, P[s, i].B, tf, tb) if r != 0 and r != L - 1]
		)
		if i == 0 or j == L - 1:
			n -= 1
		layers.append(n)
		recomp.append(nrecomp)
		i = j + 1
		s += 1

	plt.figure(figsize=(12, 6))
	plt.bar(range(p), layers, color="slateblue", edgecolor="black", linewidth=1.2)
	plt.bar(range(p), recomp, color="salmon", edgecolor="black", linewidth=1.2)

	plt.legend(["Total", "Recomputed"], fontsize=10, loc="upper right")
	plt.xlabel("Stage", fontsize=12, fontweight="bold")
	plt.ylabel("Number of layers", fontsize=12, fontweight="bold")
	# plt.title("46 blocks, 4 GPUs", fontsize=14, fontweight="bold")
	plt.grid(axis="y", alpha=0.3, linestyle="--")
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
