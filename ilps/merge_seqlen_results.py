#!/usr/bin/env python3

import argparse
import json
import os
import glob
import re
from pathlib import Path


def extract_seqlen(filename):
	"""Extract sequence length from filename."""
	match = re.search(r"seqlen_(\d+)", filename)
	if match:
		return int(match.group(1))
	return None


def merge_benchmark_results(results_dir, output_file):
	"""
	Merge all sequence length benchmark results into one file.

	Args:
	    results_dir: Directory containing benchmark results
	    output_file: Path to save the merged results
	"""
	# Find all benchmark result files
	result_files = glob.glob(os.path.join(results_dir, "bench-ilps-*_seqlen_*.json"))

	if not result_files:
		print(f"No benchmark result files found in {results_dir}")
		return False

	full_data = {}

	# Extract data from each file
	for file_path in sorted(result_files, key=extract_seqlen):
		seqlen = extract_seqlen(file_path)
		if seqlen is None:
			continue

		try:
			with open(file_path, "r") as f:
				data = json.load(f)
				data = next(iter(data.values()))
				full_data[str(seqlen)] = data
				print(f"Merged data from {file_path}")
		except (json.JSONDecodeError, FileNotFoundError) as e:
			print(f"Error loading {file_path}: {e}")

	# Save merged results
	with open(output_file, "w") as f:
		json.dump(full_data, f, indent=2)

	print(f"Merged {len(full_data)} benchmark results into {output_file}")
	return True


def main():
	parser = argparse.ArgumentParser(description="Merge sequence length benchmark results")
	parser.add_argument(
		"--results_dir",
		default="results/seqlen_benchmark",
		help="Directory containing benchmark results",
	)
	parser.add_argument(
		"--output_file",
		default="results/merged_seqlen_results.json",
		help="Path to save the merged results",
	)

	args = parser.parse_args()

	# Create output directory if it doesn't exist
	output_dir = os.path.dirname(args.output_file)
	Path(output_dir).mkdir(exist_ok=True, parents=True)

	# Merge results
	success = merge_benchmark_results(args.results_dir, args.output_file)

	if success:
		print(f"\nSuccessfully merged all sequence length benchmark results to: {args.output_file}")
		return 0
	else:
		print("\nFailed to merge results. Please check the input directory.")
		return 1


if __name__ == "__main__":
	exit(main())
