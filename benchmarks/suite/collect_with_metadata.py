import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def get_git_metadata():
	"""Collect git metadata for the current state."""
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
			"dirty_files": is_dirty if is_dirty else None,
		}
	except subprocess.CalledProcessError:
		return {
			"commit_hash": "unknown",
			"commit_date": "unknown",
			"commit_message": "unknown",
			"branch": "unknown",
			"is_dirty": False,
			"dirty_files": None,
		}


def add_metadata_to_results(results_file: Path, output_file: Path = None):
	"""Add git metadata to existing benchmark results."""
	if not results_file.exists():
		print(f"Error: Results file not found: {results_file}")
		return False

	with open(results_file, "r") as f:
		data = json.load(f)

	git_metadata = get_git_metadata()
	data["git_metadata"] = git_metadata
	data["collection_timestamp"] = datetime.now().isoformat()

	if output_file is None:
		output_file = results_file.parent / f"{results_file.stem}_with_metadata{results_file.suffix}"

	with open(output_file, "w") as f:
		json.dump(data, f, indent=2)

	print(f"Added metadata to {results_file}")
	print(f"Saved to: {output_file}")
	print("\nGit Info:")
	print(f"  Commit: {git_metadata['commit_hash'][:8]}")
	print(f"  Date: {git_metadata['commit_date']}")
	print(f"  Message: {git_metadata['commit_message']}")
	print(f"  Branch: {git_metadata['branch']}")
	if git_metadata["is_dirty"]:
		print("  WARNING: Working directory has uncommitted changes!")

	return True


def main():
	parser = argparse.ArgumentParser(
		description="Add git metadata to benchmark results for tracking performance evolution"
	)
	parser.add_argument(
		"--results-file",
		type=str,
		default="results/benchmark_suite_results.json",
		help="Path to benchmark results file",
	)
	parser.add_argument(
		"--output", type=str, help="Output file path (default: <results_file>_with_metadata.json)"
	)

	args = parser.parse_args()

	results_file = Path(args.results_file)
	output_file = Path(args.output) if args.output else None

	success = add_metadata_to_results(results_file, output_file)
	if not success:
		exit(1)


if __name__ == "__main__":
	main()
