#!/usr/bin/env python3
"""
Aggregate ILPS benchmark results across multiple repeated runs.

Each individual benchmark run stores its results in a JSON file produced by
`benchmarks/ilps_guided_benchmark.py` of the following shape::

    {
        "<n_blocks>": {
            "<solution_type>": {
                "time": <float>,            # iteration time in seconds
                "peak_mems": [<float>, ...] # list of peak GPU memories (GB) per rank
            },
            ...
        },
        ...
    }

This script globs multiple benchmark result files and merges *all* individual
measurements into a single JSON without performing any aggregation or
statistics. You can then run whatever analysis you like on the raw data.

Example
-------
    python ilps/aggregate_benchmark_runs.py \
        --pattern "results/bench-ilps-myconfig-run*.json" \
        --output-file "results/bench-ilps-myconfig-stats.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Sequence


def load_json(path: str) -> Dict[str, Any]:
	"""Safely load a JSON file, returning an empty dict on failure."""
	try:
		with open(path, "r") as f:
			return json.load(f)
	except (FileNotFoundError, json.JSONDecodeError) as exc:
		print(f"[WARN] Could not load {path}: {exc}")
		return {}


def aggregate(files: Sequence[str]) -> Dict[str, Dict[str, Any]]:
	"""Merge raw benchmark results across *files* without aggregation."""
	combined: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}

	for path in files:
		data = load_json(path)
		if not data:
			continue
		for n, sol_dict in data.items():
			for sol_type, metrics in sol_dict.items():
				entry = combined.setdefault(n, {}).setdefault(sol_type, {"times": [], "peak_mems": []})
				entry["times"].append(metrics.get("time"))
				peaks = metrics.get("peak_mems", [])
				while len(entry["peak_mems"]) < len(peaks):
					entry["peak_mems"].append([])
				for idx, val in enumerate(peaks):
					entry["peak_mems"][idx].append(val)

	return combined


def main() -> None:
	parser = argparse.ArgumentParser(description="Aggregate repeated ILPS benchmark runs")
	parser.add_argument(
		"--pattern", required=True, help="Glob pattern matching individual benchmark result JSON files"
	)
	parser.add_argument(
		"--output-file", required=True, help="Destination JSON file to write merged results"
	)

	args = parser.parse_args()
	files = sorted(glob.glob(args.pattern))

	if not files:
		print(f"No files matched pattern: {args.pattern}")
		return

	print(f"Merging {len(files)} benchmark result files …")
	aggregated = aggregate(files)

	os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
	with open(args.output_file, "w") as f:
		json.dump(aggregated, f, indent=2)
	print(f"Wrote aggregated statistics → {args.output_file}")


if __name__ == "__main__":
	main()
