#!/usr/bin/env python3
"""Aggregate raw benchmark runs across **all** sequence-length configurations.

This script looks for files matching the pattern
  results/seqlen_benchmarks/bench-ilps-* _run*.json
and merges them into a single JSON file keyed by sequence length (the value
extracted from the filename).  No statistics are computed – all raw iteration
times / peak memories are preserved.

The output structure is:

{
  "<seqlen>": {
    "<n_blocks>": {
      "<solution_type>": {
        "times": [ ... ],
        "peak_mems": [[...], ...]
      },
      ...
    },
    ...
  },
  ...
}

Example
-------
python ilps/aggregate_seqlen_runs.py \
    --results-dir results/seqlen_benchmarks \
    --output-file results/bench-ilps-seqlen-raw.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


def extract_seqlen(filename: str) -> str | None:
	match = re.search(r"seqlen_(\d+)", filename)
	return match.group(1) if match else None


def load_json(path: str) -> Dict[str, Any]:
	try:
		with open(path, "r") as f:
			return json.load(f)
	except (FileNotFoundError, json.JSONDecodeError) as exc:
		print(f"[WARN] Could not load {path}: {exc}")
		return {}


def merge_runs(files: List[str]) -> Dict[str, Any]:
	combined: Dict[str, Any] = {}

	for path in files:
		seqlen = extract_seqlen(os.path.basename(path))
		if seqlen is None:
			continue

		data = load_json(path)
		if not data:
			continue

		dest_seqlen = combined.setdefault(seqlen, {})  # type: ignore[assignment]

		# data has structure n -> solution_type -> metrics
		for n, sol_dict in data.items():
			for sol_type, metrics in sol_dict.items():
				dest_sol = dest_seqlen.setdefault(sol_type, {"times": [], "peak_mems": []})

				# merge times
				if "times" in metrics:
					dest_sol["times"].extend(metrics["times"])
				elif "time" in metrics:
					dest_sol["times"].append(metrics["time"])

				# merge peak_mems (list of lists)
				peaks = metrics.get("peak_mems", [])
				while len(dest_sol["peak_mems"]) < len(peaks):
					dest_sol["peak_mems"].append([])
				for idx, val in enumerate(peaks):
					dest_sol["peak_mems"][idx].extend(val if isinstance(val, list) else [val])

	return combined


def main() -> None:
	parser = argparse.ArgumentParser(description="Aggregate all sequence-length benchmark runs (raw)")
	parser.add_argument(
		"--results-dir", default="results/seqlen_benchmarks", help="Directory containing run JSON files"
	)
	parser.add_argument(
		"--output-file", default="results/bench-ilps-seqlen-raw.json", help="Path to write merged JSON"
	)
	args = parser.parse_args()

	pattern = os.path.join(args.results_dir, "bench-ilps-*-run*.json")
	files = glob.glob(pattern)
	if not files:
		print(f"No run files found under {args.results_dir}")
		return

	print(f"Found {len(files)} run JSON files. Merging …")
	merged = merge_runs(files)

	Path(os.path.dirname(args.output_file)).mkdir(parents=True, exist_ok=True)
	with open(args.output_file, "w") as f:
		json.dump(merged, f, indent=2)

	print(f"Wrote merged results → {args.output_file}")


if __name__ == "__main__":
	main()
