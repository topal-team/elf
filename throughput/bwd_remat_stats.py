# Script to analyze how backward remat affects performance in actual benchmarks
# 2 measurements:
# 1 - In how many ZB points ILP-FB > ILP-F, same schedule (significantly), IFF ILP-F is feasible
# 2 - In how many ZB points ILP-FB > ILP-F OR ILP-F is not feasible and ILP-FB is, same schedule
# 3 - In how many points an ILP-FB is better than all other (doable) ILP-F solutions
import os
import json
import glob
import argparse

import scipy.stats

import numpy as np


def parse_args():
	parser = argparse.ArgumentParser(
		description="Statistics on the benefit of backward rematerialization (Section 4.4)"
	)
	parser.add_argument("filenames", nargs="+", help="Paths to result JSON files (supports globs)")
	return parser.parse_args()


def expand_paths(patterns):
	paths = []
	for p in patterns:
		matches = glob.glob(p)
		if not matches and os.path.exists(p):
			matches = [p]
		paths.extend(matches)
	return sorted(set(paths))


def _is_under(a, b, alpha=0.05):
	a = np.asarray(a, dtype=float)
	b = np.asarray(b, dtype=float)

	stat, p_two = scipy.stats.mannwhitneyu(a, b, alternative="two-sided")
	p_one = p_two / 2.0 if np.median(a) < np.median(b) else 1.0 - p_two / 2.0

	return (np.median(a) < np.median(b)) and (p_one < alpha)
	# return np.median(a) < np.median(b)


def main():
	args = parse_args()

	files = expand_paths(args.filenames)
	one = [0, 0]
	two = [0, 0]
	three = [0, 0]
	notfeasible = 0
	for file in files:
		with open(file, "r") as f:
			data = json.load(f)

		for sched in ("zbh2-1W", "zbv-2W"):
			fb = data[f"FB-StageRemat-{sched}"]["all_times"]
			if f"F-StageRemat-{sched}" in data:
				fonly = data[f"F-StageRemat-{sched}"]["all_times"]
				if _is_under(fb, fonly):
					one[0] += 1
					two[0] += 1

				one[1] += 1
			else:
				notfeasible += 1
				two[0] += 1
			two[1] += 1

			if all(
				_is_under(fb, data[method]["all_times"])
				for method in data.keys()
				if method.startswith("F-StageRemat")
			):
				best = min(
					np.median(data[method]["all_times"])
					for method in data.keys()
					if method.startswith("F-StageRemat")
				)
				print(f"Speedup: {100 * (1 / np.median(fb) - 1 / best) * best:.2f}% ({file})")
				three[0] += 1
		three[1] += 1

	print(f"BR improves raw performance: {one[0]} / {one[1]} = {100 * one[0] / one[1]:.2f}%")
	print(
		f"BR improves raw performance OR makes it feasible: {two[0]} / {two[1]} = {100 * two[0] / two[1]:.2f}%"
	)
	print(
		f"BR improves overall performance: {three[0]} / {three[1]} = {100 * three[0] / three[1]:.2f}%"
	)
	print(f"Not feasible without BR: {notfeasible} / {two[1]} = {100 * notfeasible / two[1]:.2f}%")


if __name__ == "__main__":
	main()
