import json
import glob
import argparse
import sys
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List, Tuple

# ── Configuration ────────────────────────────────────────────────────────────
# Edit these constants to control which methods and schedulers appear.

ALLOWED_SCHEDULERS = {"zbh2", "zbv"}

ZB_METHODS = {"FB-StageRemat", "F-StageRemat"}
NON_ZB_METHODS = {"F-StageRemat", "F-Greedy"}

METHOD_LABELS = {
	"Uni-FB-Remat": "FullRemat",
	"Uni-F-Remat": "FullRemat",
	"FB-Greedy": "Greedy (F + B)",
	"F-Greedy": "Greedy (F)",
	"FB-StageRemat": "ILP (F + B)",
	"F-StageRemat": "ILP (F)",
	"NoRemat": "NoRemat",
	"AdaPipe": "AdaPipe",
}

# LEGEND_ORDER = ["FullRemat", "NoRemat", "Greedy", "AdaPipe", "ILP (ours)"]
LEGEND_ORDER = ["ILP (F)", "ILP (F + B)"]

SCHEDULER_ORDER = ["1F1B", "ZBH2", "MEGATRON", "ZBV"]

SCHEDULER_LABELS = {
	"1F1B": "1F1B",
	"MEGATRON": "1F1B-2W",
	# "ZBH1": "ZB-H1",
	"ZBH2": "ZB-H2",
	"ZBV": "ZB-V",
}

# COLOR_GROUPS = {
# 	"StageRemat": {"F-StageRemat", "FB-StageRemat"},
# 	"Greedy": {"F-Greedy", "FB-Greedy"},
# 	"UniRemat": {"Uni-F-Remat", "Uni-FB-Remat"},
# 	"NoRemat": {"NoRemat"},
# 	"AdaPipe": {"AdaPipe"},
# }


COLOR_GROUPS = {"F": {"F"}, "FB": {"FB"}}

PALETTE = [
	"#4E79A7",
	"#F28E2B",
	"#E15759",
	"#76B7B2",
	"#59A14F",
	"#EDC948",
	"#B07AA1",
	"#FF9DA7",
	"#9C755F",
	"#BAB0AC",
]

# method -> {config_idx: value}
MetricData = Dict[str, Dict[int, float]]


# ── CLI ──────────────────────────────────────────────────────────────────────


def get_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Plot ILPS benchmark results as vertical bar chart")
	parser.add_argument("filenames", nargs="+", help="Paths to result JSON files (supports globs)")
	parser.add_argument(
		"--output", type=str, default=None, help="Save plot to file (show if omitted)"
	)
	parser.add_argument(
		"--metric",
		choices=["time", "mfu", "throughput"],
		default="time",
		help="Metric: execution time, MFU, or throughput (sequences/s)",
	)
	parser.add_argument(
		"--sort-by",
		choices=["avg-time", "config"],
		default="avg-time",
		help="Config ordering on X axis",
	)
	parser.add_argument(
		"--separate-legend", action="store_true", help="Save legend as a separate image file"
	)
	parser.add_argument("--err", action="store_true", help="Add error bars")
	return parser.parse_args()


# ── Name helpers ─────────────────────────────────────────────────────────────


def config_label(filepath: str) -> str:
	name = os.path.splitext(os.path.basename(filepath))[0]
	if m := re.search(r"s(\d+)", name):
		return m.group(1)
	return name


def extract_scheduler(method: str) -> str:
	parts = method.split("-")
	return parts[-2] if parts[-1].endswith("W") else parts[-1]


def extract_method_type(method: str) -> str:
	parts = method.split("-")
	return "-".join(parts[:-2]) if len(parts) >= 3 else method


def extract_remat_type(method: str) -> str:
	parts = method.split("-")
	return parts[0]


def display_label(method: str) -> str:
	return METHOD_LABELS.get(extract_method_type(method), extract_method_type(method))


def scheduler_label(scheduler: str) -> str:
	return SCHEDULER_LABELS.get(scheduler.upper(), scheduler.upper())


def color_group(method: str) -> str:
	mt = extract_remat_type(method)
	for group, members in COLOR_GROUPS.items():
		if mt in members:
			return group
	return mt


def should_keep(method: str) -> bool:
	sched = extract_scheduler(method).lower()
	if sched not in ALLOWED_SCHEDULERS:
		return False
	allowed = ZB_METHODS if sched in {"zbh2", "zbv"} else NON_ZB_METHODS
	return extract_method_type(method) in allowed


# ── Data loading ─────────────────────────────────────────────────────────────


def expand_paths(patterns: List[str]) -> List[str]:
	paths: List[str] = []
	for p in patterns:
		matches = glob.glob(p)
		if not matches and os.path.exists(p):
			matches = [p]
		paths.extend(matches)
	return sorted(set(paths))


def load_results(
	file_paths: List[str],
) -> Tuple[List[str], MetricData, MetricData, MetricData, MetricData]:
	"""Load benchmark JSONs. Returns (config_labels, time_data, mfu_data, throughput_data, iqr_data)."""
	labels = [config_label(p) for p in file_paths]
	time_data: MetricData = {}
	mfu_data: MetricData = {}
	throughput_data: MetricData = {}
	iqr_data: MetricData = {}

	for idx, path in enumerate(file_paths):
		with open(path) as f:
			raw = json.load(f)
		for method, result in raw.items():
			if not should_keep(method) or not isinstance(result, dict):
				continue

			t = _extract_time(result)
			all_times = sorted(result.get("all_times", []))
			if t is not None and all_times:
				std = np.std(all_times) / t
				if std > 0.05:
					print(f"Method {method} has high std: {std * 100:.2f}%")
				iqr_data.setdefault(method, {})[idx] = np.abs(
					(np.percentile(all_times, [25, 75]) - t).reshape(2, -1) / t
				)

			if t is not None and t > 0:
				time_data.setdefault(method, {})[idx] = t
			if "mfu" in result and result["mfu"] is not None:
				try:
					mfu_data.setdefault(method, {})[idx] = float(result["mfu"])
				except (TypeError, ValueError):
					pass
			if "throughput_sps" in result and result["throughput_sps"] is not None:
				try:
					throughput_data.setdefault(method, {})[idx] = float(result["throughput_sps"])
				except (TypeError, ValueError):
					pass

	return labels, time_data, mfu_data, throughput_data, iqr_data


def _extract_time(result: dict) -> float | None:
	if "time" in result and result["time"] is not None:
		return float(result["time"])
	return None


# ── Sorting & reindexing ────────────────────────────────────────────────────


def compute_sort_order(labels: List[str], time_data: MetricData, sort_by: str) -> List[int]:
	n = len(labels)
	if sort_by == "config":
		return list(range(n))
	avgs = []
	for i in range(n):
		vals = [d[i] for d in time_data.values() if i in d]
		avgs.append(float(np.mean(vals)) if vals else float("inf"))
	return sorted(range(n), key=lambda i: avgs[i])


def reindex(labels: List[str], data: MetricData, order: List[int]) -> Tuple[List[str], MetricData]:
	old_to_new = {old: new for new, old in enumerate(order)}
	return (
		[labels[i] for i in order],
		{m: {old_to_new[k]: v for k, v in vals.items()} for m, vals in data.items()},
	)


# ── Bar layout ───────────────────────────────────────────────────────────────


def order_methods(methods: List[str]) -> List[str]:
	"""Group by scheduler, sort by LEGEND_ORDER within each group."""
	rank = {name: i for i, name in enumerate(LEGEND_ORDER)}

	def sort_key(m: str) -> Tuple[int, str]:
		return (rank.get(display_label(m), len(LEGEND_ORDER)), m)

	sched_rank = {s.lower(): i for i, s in enumerate(SCHEDULER_ORDER)}
	present = {extract_scheduler(m) for m in methods}
	scheds = sorted(present, key=lambda s: sched_rank.get(s.lower(), len(SCHEDULER_ORDER)))

	ordered: List[str] = []
	for sched in scheds:
		group = sorted([m for m in methods if extract_scheduler(m) == sched], key=sort_key)
		ordered.extend(group)
	return ordered


def compute_bar_layout(
	methods: List[str], cluster_span: float, scheduler_gap: float
) -> Tuple[float, List[float]]:
	"""Compute bar width and centered x-offsets for grouped vertical bars."""
	groups: List[int] = []
	prev = None
	for m in methods:
		s = extract_scheduler(m)
		if s != prev:
			if prev is not None:
				groups.append(0)
			prev = s
			groups.append(1)
		else:
			groups[-1] += 1

	total_bars = sum(groups)
	total_gap = max(0, len(groups) - 1) * scheduler_gap
	bar_h = (cluster_span - total_gap) / total_bars if total_bars else 0.0

	offsets: List[float] = []
	cursor = 0.0
	for gi, count in enumerate(groups):
		for _ in range(count):
			offsets.append(cursor + bar_h / 2)
			cursor += bar_h
		if gi < len(groups) - 1:
			cursor += scheduler_gap

	center = cluster_span / 2
	return bar_h, [o - center for o in offsets]


def scheduler_group_centers(methods: List[str], offsets: List[float]) -> List[Tuple[str, float]]:
	"""Center offset per scheduler group, for x-axis annotations."""
	centers: List[Tuple[str, float]] = []
	if len(methods) == 0:
		return centers
	cur = extract_scheduler(methods[0])
	start = 0
	for i in range(len(methods) + 1):
		if i == len(methods) or extract_scheduler(methods[i]) != cur:
			centers.append((cur, float(np.mean(offsets[start:i]))))
			if i < len(methods):
				cur = extract_scheduler(methods[i])
				start = i
	return centers


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_bars(
	labels: List[str],
	data: MetricData,
	iqr_data: MetricData,
	metric_label: str,
	output: str | None,
	separate_legend: bool = False,
	higher_is_better: bool = True,
	err: bool = False,
) -> None:
	plt.style.use("seaborn-v0_8")
	for k, v in {
		"figure.facecolor": "white",
		"axes.facecolor": "white",
		"axes.edgecolor": "black",
		"axes.linewidth": 0.8,
		"axes.labelcolor": "black",
		"xtick.color": "black",
		"ytick.color": "black",
		"text.color": "black",
	}.items():
		plt.rcParams[k] = v

	methods = order_methods(list(data.keys()))
	n_configs = len(labels)

	groups_seen: List[str] = []
	for m in methods:
		g = color_group(m)
		if g not in groups_seen:
			groups_seen.append(g)
	group_to_color = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups_seen)}

	# Bar positions within each subplot (shared layout across subplots)
	bar_positions: List[float] = []
	sched_group_ticks: List[Tuple[str, float]] = []
	sched_separators: List[float] = []
	pos = 0.0
	bar_width = 0.7
	sched_gap = 1.2
	prev_sched = None
	group_start_idx = 0

	for i, m in enumerate(methods):
		sched = extract_scheduler(m)
		if prev_sched is not None and sched != prev_sched:
			center = float(np.mean(bar_positions[group_start_idx:]))
			sched_group_ticks.append((scheduler_label(prev_sched), center + bar_width))
			sched_separators.append(pos - 0.5 + sched_gap / 2)
			pos += sched_gap
			group_start_idx = len(bar_positions)
		bar_positions.append(pos)
		pos += 1
		prev_sched = sched

	if prev_sched is not None and bar_positions:
		center = float(np.mean(bar_positions[group_start_idx:]))
		sched_group_ticks.append((scheduler_label(prev_sched), center))

	ncols = min(n_configs, 4)
	nrows = (n_configs + ncols - 1) // ncols
	fig, axes_arr = plt.subplots(
		nrows, ncols, figsize=(2.5 * ncols, 5 * nrows), sharey=False, squeeze=False
	)
	axes = [axes_arr[i // ncols][i % ncols] for i in range(n_configs)]
	for i in range(n_configs, nrows * ncols):
		axes_arr[i // ncols][i % ncols].set_visible(False)

	labels_seen: set[str] = set()

	for cfg_idx, (label, ax) in enumerate(zip(labels, axes)):
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)

		heights = [data[m].get(cfg_idx, np.nan) for m in methods]

		valid = [(i, h) for i, h in enumerate(heights) if not np.isnan(h)]
		best_idx = None
		if valid:
			best_idx = (max if higher_is_better else min)(valid, key=lambda x: x[1])[0]

		for m_idx, (method, h) in enumerate(zip(methods, heights)):
			lbl = display_label(method)
			if cfg_idx == 0:
				use_label = lbl if lbl not in labels_seen else "_nolegend_"
				labels_seen.add(lbl)
			else:
				use_label = "_nolegend_"

			if np.isnan(h):
				continue

			iqr = iqr_data.get(method, {}).get(cfg_idx, None)
			ax.bar(
				bar_positions[m_idx],
				h,
				width=bar_width,
				color=group_to_color[color_group(method)],
				label=use_label,
				alpha=0.9,
				edgecolor="black",
				linewidth=0.6,
				yerr=iqr * h if err and iqr is not None else None,
				error_kw={"elinewidth": 1, "capsize": 2, "capthick": 1},
			)

		if best_idx is not None:
			ax.bar(
				bar_positions[best_idx],
				heights[best_idx],
				width=bar_width,
				color="none",
				edgecolor="black",
				linewidth=3,
				zorder=5,
				label="Best" if cfg_idx == 0 else "_nolegend_",
			)

		tick_positions = [t[1] for t in sched_group_ticks]
		tick_labels_list = [t[0] for t in sched_group_ticks]
		ax.set_xticks(tick_positions)
		ax.set_xticklabels(tick_labels_list, rotation=45, ha="right", fontsize=16)

		for sep_x in sched_separators:
			ax.axvline(sep_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

		ax.set_ylim(bottom=0)
		margin = sched_gap
		if bar_positions:
			ax.set_xlim(bar_positions[0] - margin, bar_positions[-1] + margin)
		ax.set_title(f"Seqlen = {label}", fontsize=16)
		if cfg_idx % ncols == 0:
			ax.set_ylabel(metric_label, fontsize=16)
		ax.tick_params(labelsize=16)
		ax.grid(True, axis="y", alpha=0.3)

	all_handles, all_lbls = [], []
	for ax in axes:
		h, la = ax.get_legend_handles_labels()
		for hi, li in zip(h, la):
			if li not in all_lbls and li != "_nolegend_":
				all_handles.append(hi)
				all_lbls.append(li)

	rank = {name: i for i, name in enumerate(LEGEND_ORDER + ["Best"])}
	pairs = sorted(zip(all_handles, all_lbls), key=lambda x: rank.get(x[1], len(rank)))
	sorted_handles, sorted_lbls = (list(z) for z in zip(*pairs)) if pairs else ([], [])

	if err and any(iqr_data.values()):
		error_handle = plt.Line2D(
			[0], [0], color="black", linewidth=1, marker="|", markersize=8, markeredgewidth=1, label="IQR"
		)
		sorted_handles.append(error_handle)
		sorted_lbls.append("IQR")

	legend = fig.legend(
		sorted_handles,
		sorted_lbls,
		fontsize=16,
		loc="upper center",
		ncol=len(sorted_handles),
		bbox_to_anchor=(0.5, 1.0),
	)
	plt.tight_layout(rect=[0, 0, 1, 0.85])

	if output:
		if separate_legend:
			legend.remove()
		fig.savefig(output, bbox_inches="tight", dpi=300)
		if separate_legend:
			base, ext = output.rsplit(".", 1)
			fig_leg = plt.figure(figsize=(4, len(sorted_handles) * 0.3 + 0.5))
			fig_leg.legend(
				sorted_handles,
				sorted_lbls,
				loc="center",
				fontsize=16,
				frameon=False,
				ncol=len(sorted_handles),
			)
			fig_leg.savefig(f"{base}-legend.{ext}", bbox_inches="tight", dpi=300)
			plt.close(fig_leg)
	else:
		plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
	args = get_args()
	file_paths = expand_paths(args.filenames)
	if not file_paths:
		print("Error: No input files found.")
		sys.exit(1)

	labels, time_data, mfu_data, throughput_data, iqr_data = load_results(file_paths)
	order = compute_sort_order(labels, time_data, args.sort_by)

	if args.metric == "mfu":
		labels, data = reindex(labels, mfu_data, order)
		metric_label = "MFU (%) ↑"
		higher_is_better = True
	elif args.metric == "throughput":
		labels, data = reindex(labels, throughput_data, order)
		metric_label = "Throughput (sequences/s) ↑"
		higher_is_better = True
	else:
		labels, data = reindex(labels, time_data, order)
		metric_label = "Execution Time (s) ↓"
		higher_is_better = False

	plot_bars(
		labels,
		data,
		iqr_data,
		metric_label,
		args.output,
		args.separate_legend,
		higher_is_better,
		args.err,
	)


if __name__ == "__main__":
	main()
