#!/usr/bin/env python3
"""
Compare objective values between two experiment directories (baseline vs constrained).

Each directory should contain JSON files for different methods, as produced by
`simulate.py`. The script plots:
  - Overlaid curves: solid for baseline, dashed for constrained variant
  - A relative overhead subplot showing the % increase due to the constraint

Usage:
  python plot_uniform.py results/fig6/per-microbatch/ngpus16 results/fig6/uniform/ngpus16 \
	  --no-overhead --labels "Per-Mb" "Uniform" --throughput --nmb 32 --output fig6.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib.ticker import FuncFormatter, FixedLocator

SCHEDULES_TO_LABEL = {
	"1f1b": "1F1B",
	# "megatron": "1F1B-2W",
	"zbv": "ZB-V",
	# "zbh2": "ZB-H2",
}
METHODS_TO_LABEL = {"F-StageRemat": "ILP", "FB-StageRemat": "ILP"}


def method_to_label(method: str) -> str:
	*method_parts, scheduler, nwaves = method.split("-")
	scheduler = SCHEDULES_TO_LABEL.get(scheduler, scheduler.upper())
	return f"{scheduler}"


def _load_method_curves(directory: str) -> Dict[str, Tuple[List[int], List[float]]]:
	"""Load curves keyed by method label.

	Returns a dict mapping method_label -> (memories_mb_asc, makespan_s_asc).
	"""
	curves: Dict[str, Tuple[List[int], List[float]]] = {}

	for fname in sorted(os.listdir(directory)):
		if not fname.endswith(".json"):
			continue
		if fname.startswith("profiled"):
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
		except Exception as exc:
			print(f"Skipping {fname}: failed to load JSON ({exc})")
			continue

		mem_to_obj_ms: List[Tuple[int, float]] = []
		for mem_key, solution in data.items():
			try:
				mem_mb = int(mem_key)
				objective_ms = float(solution["objective"])
			except (KeyError, ValueError, TypeError):
				continue
			mem_to_obj_ms.append((mem_mb, objective_ms))

		if not mem_to_obj_ms:
			continue

		mem_to_obj_ms.sort(key=lambda t: t[0])
		memories_mb = [m for (m, _) in mem_to_obj_ms]
		makespan_s = [ms / 1_000.0 for (_, ms) in mem_to_obj_ms]

		label = method_to_label(method_name)
		curves[label] = (memories_mb, makespan_s)

	return curves


def _set_publication_style() -> None:
	"""Configure a clean, publication-ready Matplotlib style.

	- Colorblind-friendly Okabe–Ito palette
	- Subtle grid, de-emphasized spines
	- Consistent font sizes and line widths
	"""

	palette = sns.color_palette("deep")
	palette.pop(1)  # i don't like orange
	palette.pop(1)
	# palette.pop(1)

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
			"xtick.minor.size": 0.0,  # disable minor ticks
			"ytick.major.size": 3.5,
			"axes.grid.which": "major",
			"axes.grid": True,
			"grid.linestyle": ":",
		}
	)


def _gb_formatter(x: float, _pos: int | None) -> str:
	return f"{x / 1024:.1f}"


def _nice_step(raw_step: float) -> float:
	if raw_step <= 0:
		return 1.0
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
	start = math.ceil(min_gb / step) * step
	end = math.floor(max_gb / step) * step
	ticks_gb: List[float] = []
	val = start
	while val <= end + 1e-9 and len(ticks_gb) < 100:
		ticks_gb.append(round(val, 10))
		val += step
	return [gb * 1024.0 for gb in ticks_gb]


def _pad_minmax_values(
	curves: Dict[str, Tuple[List[int], List[float]]], global_min: int, global_max: int
) -> Dict[str, Tuple[List[int], List[float]]]:
	"""Extend each curve to global min/max memory by repeating boundary values."""
	padded: Dict[str, Tuple[List[int], List[float]]] = {}
	for label, (mems, ys) in curves.items():
		if not mems:
			continue
		new_mems: List[int] = []
		new_ys: List[float] = []

		if global_min < mems[0]:
			new_mems.append(global_min)
			new_ys.append(ys[0])

		new_mems.extend(mems)
		new_ys.extend(ys)

		if global_max > mems[-1]:
			new_mems.append(global_max)
			new_ys.append(ys[-1])

		padded[label] = (new_mems, new_ys)
	return padded


def plot_ablation(
	dir_baseline: str,
	dir_constrained: str,
	label_baseline: str,
	label_constrained: str,
	output_path: str | None,
	show_overhead: bool,
	throughput: bool = False,
	nmb: int | None = None,
) -> None:
	_set_publication_style()
	# sns.set_context(
	#     "poster", font_scale=0.6, rc={"lines.linewidth": 2.0, "lines.markersize": 5}
	# )

	curves_base = _load_method_curves(dir_baseline)
	curves_cons = _load_method_curves(dir_constrained)

	if not curves_base and not curves_cons:
		raise SystemExit("No method JSON files found in either directory.")

	all_mems_global: List[int] = []
	for mems, _ in curves_base.values():
		all_mems_global.extend(mems)
	for mems, _ in curves_cons.values():
		all_mems_global.extend(mems)

	if all_mems_global:
		global_min = min(all_mems_global)
		global_max = max(all_mems_global)
		curves_base = _pad_minmax_values(curves_base, global_min, global_max)
		curves_cons = _pad_minmax_values(curves_cons, global_min, global_max)

	all_methods = sorted(set(curves_base.keys()) | set(curves_cons.keys()))

	if not all_methods:
		raise SystemExit("No methods to plot.")

	ax_main = plt.gca()

	all_mems_mb: List[int] = []

	colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])

	for method in all_methods:
		color = next(colors)

		if method in curves_base:
			mems, ys = curves_base[method]
			if throughput:
				ys = [nmb / y for y in ys]
			all_mems_mb.extend(mems)
			ax_main.plot(
				mems,
				ys,
				marker="o",
				linestyle="-",
				markerfacecolor="white",
				markeredgewidth=0.8,
				label=f"{method} ({label_baseline})",
				color=color,
			)

		if method in curves_cons:
			mems, ys = curves_cons[method]
			if throughput:
				ys = [nmb / y for y in ys]
			all_mems_mb.extend(mems)
			ax_main.plot(
				mems,
				ys,
				marker="o",
				linestyle="--",
				markerfacecolor="white",
				markeredgewidth=0.8,
				label=f"{method} ({label_constrained})",
				color=color,
			)

	ax_main.set_ylabel("Throughput (seq/s) ↑" if throughput else "Makespan (s) ↓")

	ax_main.set_xlabel("Memory constraint (GB)")

	tick_positions_mb = _generate_gb_ticks(all_mems_mb, max_ticks=6)
	target_ax = ax_main
	if tick_positions_mb:
		target_ax.xaxis.set_major_locator(FixedLocator(tick_positions_mb))
	target_ax.xaxis.set_major_formatter(FuncFormatter(_gb_formatter))

	all_handles, all_lbls = [], []
	h, la = ax_main.get_legend_handles_labels()
	for hi, li in zip(h, la):
		if li not in all_lbls and li != "_nolegend_":
			all_handles.append(hi)
			all_lbls.append(li)

	plt.legend(
		all_handles,
		all_lbls,
		loc="best",
		# ncol=len(all_handles) // 2,
		# bbox_to_anchor=(0.5, 1.0),
		# frameon=False,
	)
	plt.tight_layout()

	if output_path is None:
		plt.show()
	else:
		plt.savefig(output_path, bbox_inches="tight", dpi=300, transparent=True)
		print(f"Saved to {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compare objective values between baseline and constrained experiments."
	)
	parser.add_argument(
		"dir_baseline", type=str, help="Path to baseline directory containing method JSON files."
	)
	parser.add_argument(
		"dir_constrained", type=str, help="Path to constrained directory containing method JSON files."
	)
	parser.add_argument(
		"--labels",
		nargs=2,
		default=["Baseline", "Constrained"],
		metavar=("BASELINE", "CONSTRAINED"),
		help="Labels for the two variants (default: 'Baseline' 'Constrained').",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Path to save figure. If omitted, displays interactively.",
	)
	parser.add_argument(
		"--no-overhead",
		action="store_true",
		default=False,
		help="Disable the relative overhead subplot.",
	)
	parser.add_argument(
		"--throughput",
		"-t",
		action="store_true",
		default=False,
		help="Plot throughput (seq/s) instead of makespan.",
	)
	parser.add_argument(
		"--nmb",
		"-m",
		type=int,
		default=None,
		help="Number of micro-batches (required for throughput mode).",
	)
	args = parser.parse_args()

	if args.throughput and args.nmb is None:
		parser.error("--nmb is required when using --throughput")

	dir_baseline = os.path.abspath(args.dir_baseline)
	dir_constrained = os.path.abspath(args.dir_constrained)

	for d in (dir_baseline, dir_constrained):
		if not os.path.isdir(d):
			raise SystemExit(f"Not a directory: {d}")

	plot_ablation(
		dir_baseline,
		dir_constrained,
		args.labels[0],
		args.labels[1],
		args.output,
		show_overhead=not args.no_overhead,
		throughput=args.throughput,
		nmb=args.nmb,
	)


if __name__ == "__main__":
	main()
