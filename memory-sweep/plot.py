#!/usr/bin/env python3
"""
Plot makespan vs memory constraint for all methods in a given ngpus directory.

The directory is expected to contain multiple JSON files, each corresponding to a
method (e.g., "F-StageRemat-1f1b.json"), as produced by
`simulate.py`. A `profiled.json` file may also be
present and will be ignored.

Usage:
  python plot.py results/fig3/ngpus8 --throughput --nmb 16 \
	  --output makespan_vs_mem.png

Notes:
  - X axis: memory constraint (MB), displayed in decreasing order
  - Y axis: makespan (seconds)
  - If a method is missing points at the beginning (largest memories), its curve
	is left-padded by repeating its first available point across those memories.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple
import seaborn as sns

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import FuncFormatter, FixedLocator

SCHEDULES_TO_LABEL = {"1f1b": "1F1B", "megatron": "1F1B-2W", "zbv": "ZB-V", "zbh2": "ZB-H2"}
METHODS_TO_LABEL = {"F-StageRemat": "ILP", "FB-StageRemat": "ILP"}


def method_to_label(method: str) -> str:
	*_, scheduler, _ = method.split("-")
	return SCHEDULES_TO_LABEL.get(scheduler, scheduler.upper())


def _load_method_curves(directory: str) -> List[Tuple[str, List[int], List[float]]]:
	"""Load curves for each method in the directory.

	Returns a list of tuples: (method_name, memories_mb_desc, makespan_s_desc)
	where memories_mb_desc is sorted in descending order.
	"""
	curves: List[Tuple[str, List[int], List[float]]] = []

	for fname in sorted(os.listdir(directory)):
		if not fname.endswith(".json"):
			continue
		if fname == "profiled.json":
			continue

		if all(schedule not in fname for schedule in SCHEDULES_TO_LABEL):
			continue

		if all(method not in fname for method in METHODS_TO_LABEL):
			continue

		method_name = os.path.splitext(fname)[0]
		for method, label in METHODS_TO_LABEL.items():
			method_name = method_name.replace(method, label)

		method_path = os.path.join(directory, fname)
		try:
			with open(method_path, "r") as f:
				data: Dict[str, Dict[str, object]] = json.load(f)
		except Exception as exc:  # noqa: BLE001
			print(f"Skipping {fname}: failed to load JSON ({exc})")
			continue

		# Keys are memory constraints (likely dumped as strings). Convert to int.
		mem_to_obj_ms: List[Tuple[int, float]] = []
		for mem_key, solution in data.items():
			try:
				mem_mb = int(mem_key)
				objective_ms = float(solution["objective"])  # milliseconds
			except (KeyError, ValueError, TypeError):
				# Skip malformed entries
				continue
			mem_to_obj_ms.append((mem_mb, objective_ms))

		if not mem_to_obj_ms:
			continue

		# Sort by memory ascending (increasing)
		mem_to_obj_ms.sort(key=lambda t: t[0], reverse=False)
		memories_mb_desc = [m for (m, _) in mem_to_obj_ms]
		makespan_s_desc = [ms / 1_000.0 for (_, ms) in mem_to_obj_ms]

		curves.append((method_name, memories_mb_desc, makespan_s_desc))

	return curves


def _left_pad_with_first_point(
	curves: List[Tuple[str, List[int], List[float]]],
) -> List[Tuple[str, List[int], List[float]]]:
	"""Extend each curve on the left with its first point if needed.

	We compute the global set of memory values and, for each method, if it lacks
	some of the largest memories that exist in other methods, we repeat its first
	available point at those memories. Internal gaps (between first and last
	memory) are not filled.
	"""
	if not curves:
		return curves

	global_memories = set()
	for _, mems, _ in curves:
		global_memories.update(mems)

	global_mems_desc = sorted(global_memories)

	padded_curves: List[Tuple[str, List[int], List[float]]] = []
	for method_name, mems_desc, ys_desc in curves:
		mem_set = set(mems_desc)
		if not mems_desc:
			continue

		# Use the largest available point for padding of larger memories
		first_mem = mems_desc[-1]
		first_y = ys_desc[-1]

		# Build new lists by left-padding with first point for any global mem > first_mem
		new_mems: List[int] = []
		new_ys: List[float] = []

		for m in global_mems_desc:
			if m in mem_set:
				# Use the actual y for this memory value
				idx = mems_desc.index(m)
				new_mems.append(m)
				new_ys.append(ys_desc[idx])
			elif m > first_mem:
				# Left-pad with first point
				new_mems.append(m)
				new_ys.append(first_y)
			# else: do not fill internal or right-side gaps

		# Keep ascending order
		padded_curves.append((method_name, new_mems, new_ys))

	return padded_curves


def _pad_minmax_values(curves):

	global_memories = set()
	for _, mems, _ in curves:
		global_memories.update(mems)

		max_mem = max(global_memories)
	padded_curves: List[Tuple[str, List[int], List[float]]] = []
	for method_name, mems_desc, ys_desc in curves:
		mem_set = set(mems_desc)
		if not mems_desc:
			continue

		last_y = ys_desc[-1]

		# Build new lists by left-padding with first point for any global mem > first_mem
		new_mems: List[int] = []
		new_ys: List[float] = []

		# if min_mem not in mem_set:
		#       new_mems.append(min_mem)
		# 	new_ys.append(first_y)

		for m in mems_desc:
			if m in mem_set:
				# Use the actual y for this memory value
				idx = mems_desc.index(m)
				new_mems.append(m)
				new_ys.append(ys_desc[idx])

		if max_mem not in mem_set:
			new_mems.append(max_mem)
			new_ys.append(last_y)

		# Keep ascending order
		padded_curves.append((method_name, new_mems, new_ys))

	return padded_curves


def _set_publication_style() -> None:
	"""Configure a clean, publication-ready Matplotlib style.

	- Colorblind-friendly Okabe–Ito palette
	- Subtle grid, de-emphasized spines
	- Consistent font sizes and line widths
	"""

	palette = sns.color_palette("deep")
	palette.pop(1)  # i don't like orange
	palette[2], palette[3] = palette[3], palette[2]

	plt.rcParams.update(
		{
			"figure.figsize": (7, 4),
			"figure.dpi": 100,
			"savefig.dpi": 300,
			"savefig.bbox": "tight",
			# "font.size": 20,
			"axes.titlesize": 20,
			"axes.labelsize": 20,
			"xtick.labelsize": 16,
			"ytick.labelsize": 16,
			"legend.fontsize": 10,
			"axes.linewidth": 0.8,
			"axes.prop_cycle": cycler(color=palette),
			"lines.linewidth": 2.0,
			# "lines.markersize": 0.0,
			"xtick.direction": "out",
			"ytick.direction": "out",
			"xtick.major.size": 3.5,
			"xtick.minor.size": 0.0,  # disable minor ticks
			"ytick.major.size": 3.5,
			"axes.grid.which": "major",
			"axes.grid": True,
			"grid.linestyle": ":",
		}
	)


def _gb_formatter(x: float, _pos: int | None) -> str:
	"""Format memory values in GB with one decimal."""
	return f"{x / 1024:.1f}"


def _nice_step(raw_step: float) -> float:
	"""Return a 'nice' step size >= raw_step using {1, 2, 5} * 10^n."""
	if raw_step <= 0:
		return 1.0
	import math

	exponent = math.floor(math.log10(raw_step))
	scaled = raw_step / (10**exponent)
	if scaled <= 1:
		nice = 1
	elif scaled <= 2:
		nice = 2
	elif scaled <= 5:
		nice = 5
	else:
		nice = 10
	return nice * (10**exponent)


def _generate_gb_ticks(all_mems_mb: List[int], max_ticks: int = 6) -> List[float]:
	"""Generate tick positions (in MB) at nice GB values within data range."""
	if not all_mems_mb:
		return []
	min_mb = min(all_mems_mb)
	max_mb = max(all_mems_mb)
	min_gb = min_mb / 1024.0
	max_gb = max_mb / 1024.0
	range_gb = max(0.0, max_gb - min_gb)
	if range_gb == 0:
		return [min_mb]
	raw_step = range_gb / max(1, max_ticks - 1)
	step = _nice_step(raw_step)
	import math

	start = math.ceil(min_gb / step) * step
	end = math.floor(max_gb / step) * step
	ticks_gb: List[float] = []
	val = start
	# Guard against infinite loops due to FP
	while val <= end + 1e-9 and len(ticks_gb) < 100:
		ticks_gb.append(round(val, 10))
		val += step
	return [gb * 1024.0 for gb in ticks_gb]


def plot_makespan_vs_memory(
	directory: str, output_path: str | None, separate_legend: bool, throughput: bool, nmb: int | None
) -> None:
	_set_publication_style()
	# sns.set_context(
	# 	"paper", font_scale=2, rc={"lines.linewidth": 2, "lines.markersize": 6.5}
	# )
	curves = _load_method_curves(directory)
	if not curves:
		raise SystemExit(f"No method JSON files found in {directory} (excluding 'profiled.json').")

	curves = _pad_minmax_values(curves)

	ax = plt.gca()
	for method_name, memories_mb_desc, makespan_s_desc in curves:
		if not memories_mb_desc:
			continue

		if throughput:
			makespan_s_desc = [nmb / ms for ms in makespan_s_desc]

		ax.plot(
			memories_mb_desc,
			makespan_s_desc,
			marker="o",
			markerfacecolor="white",
			markeredgewidth=0.8,
			label=method_to_label(method_name),
		)

	# Labels and formatting
	ax.set_xlabel("Memory constraint (GB)")
	# ax.set_ylabel("Makespan (s) ↓" if not throughput else "Throughput (seq/s) ↑")
	# ax.set_title(os.path.basename(directory))

	# Grid and spines
	# ax.grid(True, which="major")
	# for spine in ("top", "right"):
	# 	ax.spines[spine].set_visible(False)
	# for spine in ("left", "bottom"):
	# 	ax.spines[spine].set_linewidth(0.8)

	# Ticks: show increasing memory, label in GB with nice integer values
	all_mems_mb: List[int] = []
	for _, mems, _ in curves:
		all_mems_mb.extend(mems)
	tick_positions_mb = _generate_gb_ticks(all_mems_mb, max_ticks=6)
	if tick_positions_mb:
		ax.xaxis.set_major_locator(FixedLocator(tick_positions_mb))
	ax.xaxis.set_major_formatter(FuncFormatter(_gb_formatter))

	# Legend
	handles, labels = ax.get_legend_handles_labels()
	legend = ax.legend(frameon=False)
	if separate_legend:
		legend.remove()
	# Save and/or show
	if output_path is None:
		plt.show()
	else:
		plt.savefig(output_path, bbox_inches="tight", dpi=300, transparent=False)
	if separate_legend:
		legend_output = output_path.rsplit(".", 1)
		legend_filename = f"{legend_output[0]}_legend.{legend_output[1]}"

		legend_fig = plt.figure(figsize=(4, len(handles) * 0.3 + 0.5))
		legend_fig.legend(handles, labels, loc="center", frameon=False, ncol=len(labels))
		legend_fig.savefig(legend_filename, bbox_inches="tight", dpi=300, transparent=True)
		plt.close(legend_fig)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=("Plot makespan vs memory for all methods in a given ngpus directory.")
	)
	parser.add_argument(
		"directory",
		type=str,
		help=("Path to an 'ngpus*' directory containing method JSON files (excluding profiled.json)."),
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help=("Optional path to save the figure (defaults to '<directory>/makespan_vs_mem.png')."),
	)
	parser.add_argument(
		"--separate-legend",
		action="store_true",
		default=False,
		help=("Save the legend to a separate file."),
	)
	parser.add_argument(
		"--throughput",
		"-t",
		action="store_true",
		default=False,
		help=("Plot throughput instead of makespan."),
	)
	parser.add_argument(
		"--nmb",
		"-m",
		type=int,
		default=None,
		help=("Number of micro-batches to use for throughput calculation."),
	)
	args = parser.parse_args()

	assert not args.throughput or args.nmb is not None, (
		"Number of micro-batches must be specified for throughput calculation."
	)

	directory = os.path.abspath(args.directory)
	if not os.path.isdir(directory):
		raise SystemExit(f"Not a directory: {directory}")

	plot_makespan_vs_memory(directory, args.output, args.separate_legend, args.throughput, args.nmb)


if __name__ == "__main__":
	main()
