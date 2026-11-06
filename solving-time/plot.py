import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from argparse import ArgumentParser
from cycler import cycler


def parse_args():
	parser = ArgumentParser()

	parser.add_argument("data", nargs="?")
	parser.add_argument("--output", type=str, required=False, default=None)
	parser.add_argument(
		"--unit", choices=["ranks", "mbs"], default="ranks", help="Legend unit for the result keys"
	)
	return parser.parse_args()


def log_formatter(x, pos):
	# copied from: https://stackoverflow.com/a/33213196
	# Find the number of decimal places required
	decimalplaces = int(np.maximum(-np.log10(x), 0))  # =0 for numbers >=1
	# Insert that number into a format string
	formatstring = "{{:.{:1d}f}}".format(decimalplaces)
	# Return the formatted tick label
	return formatstring.format(x)


def _set_publication_style() -> None:
	"""Configure a clean, publication-ready Matplotlib style.

	- Subtle grid, de-emphasized spines
	- Consistent font sizes and line widths
	"""

	palette = sns.color_palette("deep")
	palette.pop(1)  # i don't like orange

	plt.rcParams.update(
		{
			"figure.figsize": (7, 4),
			"figure.dpi": 100,
			"savefig.dpi": 300,
			"savefig.bbox": "tight",
			# "font.size": 20,
			"axes.titlesize": 14,
			"axes.labelsize": 14,
			"xtick.labelsize": 11,
			"ytick.labelsize": 11,
			"legend.fontsize": 14,
			"axes.linewidth": 0.8,
			"axes.prop_cycle": cycler(color=palette),
			"lines.linewidth": 2.0,
			"lines.markersize": 0.0,
			"xtick.direction": "out",
			"ytick.direction": "out",
			"xtick.major.size": 3.5,
			"xtick.minor.size": 0.0,
			"ytick.major.size": 3.5,
			"axes.grid.which": "major",
			"axes.grid": True,
			"grid.linestyle": ":",
		}
	)


def main():
	args = parse_args()

	with open(args.data, "r") as f:
		data = json.load(f)

	# plt.style.use("seaborn-v0_8-whitegrid")
	_set_publication_style()
	colors = sns.color_palette("deep")
	colors.pop(1)  # i don't like orange

	for nproc in sorted(data, key=int):
		times = []
		objectives = []
		final_bound = -1
		for state in reversed(data[nproc]["progress"]):  # from last to first to find the last bound
			if state["Objective"] == "cutoff":  # weird end steps
				continue
			if final_bound == -1 and state["Objective"] == "":
				final_bound = float(state["CutsBestBound"])
			if state["Objective"] in ("", "infeasible"):  # fake improvements, or no solution found
				continue
			# prfloat(f"After {state['Time']}s, obj = {state['Objective']}, bound = {state['CutsBestBound']}, gap = {state['Gap']}")
			times.append(float(state["Time"]))
			objectives.append(float(state["BestInteger"]))

		gaps = [100 * (obj / final_bound - 1) for obj in reversed(objectives)]
		if final_bound == -1:
			gaps = [0 for _ in gaps]
		times = list(reversed(times))

		if data[nproc]["solution_status"] == 1:  # 1 is LpSolutionOptimal
			times.append(data[nproc]["elapsed"])
			gaps.append(0)
		# print(f"{nproc}: {gaps=}, {objectives=}, {final_bound=}")
		plt.plot(times, gaps, label=f"{nproc} {args.unit}")

	red = colors[2]
	blue = colors[0]
	pos = plt.gca().get_position()
	plt.text(pos.x0 - 0.095, 2, "1%", color=blue, fontsize=12)
	plt.axhline(1, linestyle="--", linewidth=1, alpha=0.7, color=blue)
	plt.text(pos.x0 - 0.095, 6, "5%", color=blue, fontsize=12)
	plt.axhline(5, linestyle="--", linewidth=1, alpha=0.7, color=blue)
	plt.text(pos.x0 + 670, 11, "10 minutes", color=red, fontsize=12)
	plt.axvline(600, ymax=1, linestyle="-.", linewidth=1.2, alpha=0.7, color=red)

	plt.legend(fontsize=14, loc="best")
	plt.xlabel("Solving time (s)", fontsize=14)
	plt.xscale("log", base=10)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.gca().xaxis.set_major_formatter(FuncFormatter(log_formatter))
	# plt.gca().set_xticks([])
	plt.ylabel("Gap (%)", fontsize=14)
	plt.tight_layout()
	if args.output:
		plt.savefig(args.output)
	else:
		plt.show()


if __name__ == "__main__":
	main()
