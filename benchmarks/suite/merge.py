import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_metadata():
	"""Collect git metadata for tracking performance evolution."""
	try:
		commit_hash = (
			subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
			.decode()
			.strip()
		)

		commit_date = (
			subprocess.check_output(
				["git", "show", "-s", "--format=%ci", "HEAD"], stderr=subprocess.DEVNULL
			)
			.decode()
			.strip()
		)

		commit_message = (
			subprocess.check_output(
				["git", "show", "-s", "--format=%s", "HEAD"], stderr=subprocess.DEVNULL
			)
			.decode()
			.strip()
		)

		branch = (
			subprocess.check_output(
				["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
			)
			.decode()
			.strip()
		)

		is_dirty = (
			subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
			.decode()
			.strip()
		)

		return {
			"commit_hash": commit_hash,
			"commit_date": commit_date,
			"commit_message": commit_message,
			"branch": branch,
			"is_dirty": bool(is_dirty),
		}
	except subprocess.CalledProcessError:
		return None


def merge_results(results_dir: str, output_filename: str, save_to_history: bool = True):
	results_path = Path(results_dir)

	if not results_path.exists():
		print(f"Error: Results directory not found: {results_dir}")
		return

	result_files = sorted(results_path.glob("benchmark_suite_results_*gpu.json"))

	if not result_files:
		print(f"Warning: No benchmark result files found in {results_dir}")
		print("Looking for files matching: benchmark_suite_results_*gpu.json")
		return

	print(f"Found {len(result_files)} result files to merge:")
	for f in result_files:
		print(f"  - {f.name}")

	all_results = []
	configs = []
	timestamps = []

	for result_file in result_files:
		with open(result_file, "r") as f:
			data = json.load(f)
			all_results.extend(data["results"])
			configs.append(data["config"])
			timestamps.append(data["timestamp"])

	merged_config = configs[0] if configs else {}

	git_metadata = get_git_metadata()

	merged_data = {
		"timestamp": datetime.now().isoformat(),
		"individual_timestamps": timestamps,
		"config": merged_config,
		"results": all_results,
	}

	if git_metadata:
		merged_data["git_metadata"] = git_metadata

	output_file = results_path / output_filename
	with open(output_file, "w") as f:
		json.dump(merged_data, f, indent=2)

	print(f"\nMerged {len(all_results)} results into: {output_file}")

	if git_metadata:
		print("\nGit metadata:")
		print(f"  Commit: {git_metadata['commit_hash'][:8]}")
		print(f"  Date: {git_metadata['commit_date']}")
		print(f"  Message: {git_metadata['commit_message']}")
		if git_metadata["is_dirty"]:
			print("  WARNING: Working directory has uncommitted changes!")

	if save_to_history and git_metadata:
		history_dir = results_path / "history"
		history_dir.mkdir(exist_ok=True)

		commit_hash_short = git_metadata["commit_hash"][:8]
		timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
		history_filename = f"bench_{timestamp_str}_{commit_hash_short}.json"
		history_file = history_dir / history_filename

		with open(history_file, "w") as f:
			json.dump(merged_data, f, indent=2)

		print(f"\nSaved to history: {history_file}")
	print("\n" + "=" * 70)
	print("MFU Summary")
	print("=" * 70)
	for result in all_results:
		model = result["model"]
		world_size = result["world_size"]

		if "mfu" in result and result["mfu"] is not None:
			mfu = result["mfu"] * 100
			if "throughput_tokens_per_sec" in result:
				throughput = f"{result['throughput_tokens_per_sec']:.2f} tokens/s"
			else:
				throughput = f"{result['throughput_samples_per_sec']:.2f} samples/s"
			max_mem = result["max_memory"]
			print(
				f"  {model:20s} @ {world_size} GPUs: MFU {mfu:5.2f}%  |  {throughput:15s}  |  {max_mem:.2f}GB"
			)
		else:
			if "throughput_tokens_per_sec" in result:
				throughput = f"{result['throughput_tokens_per_sec']:.2f} tokens/s"
			else:
				throughput = f"{result['throughput_samples_per_sec']:.2f} samples/s"
			max_mem = result["max_memory"]
			print(f"  {model:20s} @ {world_size} GPUs: {throughput:15s}  |  {max_mem:.2f}GB")
	print("=" * 70)


def main():
	parser = argparse.ArgumentParser(description="Merge benchmark suite results")
	parser.add_argument(
		"--results-dir", type=str, default="results", help="Directory containing result files"
	)
	parser.add_argument(
		"--output",
		type=str,
		default="benchmark_suite_results.json",
		help="Output filename for merged results",
	)
	parser.add_argument(
		"--no-history", action="store_true", help="Do not save results to history directory"
	)

	args = parser.parse_args()
	merge_results(args.results_dir, args.output, save_to_history=not args.no_history)


if __name__ == "__main__":
	main()
