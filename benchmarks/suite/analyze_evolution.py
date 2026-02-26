import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_benchmark_history(history_dir: Path) -> List[Dict]:
	"""Load all benchmark results with metadata from a directory."""
	results = []

	if not history_dir.exists():
		print(f"Error: History directory not found: {history_dir}")
		return results

	json_files = sorted(history_dir.glob("*.json"))

	if not json_files:
		print(f"Warning: No JSON files found in {history_dir}")
		return results

	for json_file in json_files:
		try:
			with open(json_file, "r") as f:
				data = json.load(f)

			if "git_metadata" not in data:
				print(f"Warning: {json_file.name} missing git metadata, skipping")
				continue

			results.append(data)
		except Exception as e:
			print(f"Error loading {json_file.name}: {e}")

	results.sort(key=lambda x: x.get("git_metadata", {}).get("commit_date", ""))

	print(f"Loaded {len(results)} benchmark results from {history_dir}")
	return results


def extract_metrics_by_config(history: List[Dict]) -> Dict[str, List[Tuple]]:
	"""
	Extract metrics organized by model/scale configuration.

	Returns:
		Dict mapping "model_name@Ngpu" -> [(commit_hash, commit_date, metrics_dict)]
	"""
	metrics_by_config = {}

	for entry in history:
		commit_hash = entry["git_metadata"]["commit_hash"][:8]
		commit_date = entry["git_metadata"]["commit_date"]
		commit_msg = entry["git_metadata"]["commit_message"]

		for result in entry.get("results", []):
			model = result["model"]
			world_size = result["world_size"]
			config_key = f"{model}@{world_size}gpu"

			if config_key not in metrics_by_config:
				metrics_by_config[config_key] = []

			metrics = {
				"commit_hash": commit_hash,
				"commit_date": commit_date,
				"commit_message": commit_msg,
				"throughput_samples": result.get("throughput_samples_per_sec", 0),
				"throughput_tokens": result.get("throughput_tokens_per_sec"),
				"avg_iter_time": result.get("avg_iteration_time", 0),
				"max_memory": result.get("max_memory", 0),
				"mfu": result.get("mfu"),
			}

			metrics_by_config[config_key].append(metrics)

	return metrics_by_config


def plot_evolution(metrics_by_config: Dict[str, List[Tuple]], output_dir: Path):
	"""Create visualization plots for performance evolution, combining all models."""
	output_dir.mkdir(parents=True, exist_ok=True)

	# Group by GPU count
	by_gpu_count = {}
	for config_key, metrics_list in metrics_by_config.items():
		if not metrics_list:
			continue
		# Extract GPU count from config_key (e.g., "FullTransformer@4gpu" -> 4)
		gpu_count = int(config_key.split("@")[1].replace("gpu", ""))
		if gpu_count not in by_gpu_count:
			by_gpu_count[gpu_count] = {}
		model_name = config_key.split("@")[0]
		by_gpu_count[gpu_count][model_name] = metrics_list

	# Create one plot per GPU count with all models
	colors = ["purple", "blue", "green", "orange", "red", "brown", "pink", "gray"]

	for gpu_count, models_data in by_gpu_count.items():
		fig, axes = plt.subplots(2, 2, figsize=(16, 12))
		fig.suptitle(f"Performance Evolution @ {gpu_count} GPUs", fontsize=16, fontweight="bold")

		has_mfu = False
		has_tokens = False

		# Check what data is available
		for model_name, metrics_list in models_data.items():
			if metrics_list[0]["mfu"] is not None:
				has_mfu = True
			if metrics_list[0]["throughput_tokens"] is not None:
				has_tokens = True

		# Plot 1: MFU or Throughput (samples)
		for idx, (model_name, metrics_list) in enumerate(models_data.items()):
			commits = [m["commit_hash"] for m in metrics_list]
			color = colors[idx % len(colors)]

			if has_mfu and metrics_list[0]["mfu"] is not None:
				mfu_values = [m["mfu"] * 100 for m in metrics_list]
				axes[0, 0].plot(
					range(len(commits)), mfu_values, marker="o", linewidth=2, color=color, label=model_name
				)
			else:
				throughput_samples = [m["throughput_samples"] for m in metrics_list]
				axes[0, 0].plot(
					range(len(commits)),
					throughput_samples,
					marker="o",
					linewidth=2,
					color=color,
					label=model_name,
				)

		axes[0, 0].set_xlabel("Commits")
		axes[0, 0].set_ylabel("MFU (%)" if has_mfu else "Samples/sec")
		axes[0, 0].set_title("Model FLOPs Utilization (MFU)" if has_mfu else "Throughput (Samples)")
		axes[0, 0].grid(True, alpha=0.3)
		axes[0, 0].legend()

		# Use commits from first model for x-axis labels
		first_model_commits = [m["commit_hash"] for m in list(models_data.values())[0]]
		axes[0, 0].set_xticks(range(len(first_model_commits)))
		axes[0, 0].set_xticklabels(first_model_commits, rotation=45, ha="right")

		# Plot 2: Throughput (tokens) if available
		if has_tokens:
			for idx, (model_name, metrics_list) in enumerate(models_data.items()):
				if metrics_list[0]["throughput_tokens"] is not None:
					commits = [m["commit_hash"] for m in metrics_list]
					throughput_tokens = [m["throughput_tokens"] for m in metrics_list]
					color = colors[idx % len(colors)]
					axes[0, 1].plot(
						range(len(commits)),
						throughput_tokens,
						marker="o",
						linewidth=2,
						color=color,
						label=model_name,
					)
			axes[0, 1].set_xlabel("Commits")
			axes[0, 1].set_ylabel("Tokens/sec")
			axes[0, 1].set_title("Throughput (Tokens)")
			axes[0, 1].grid(True, alpha=0.3)
			axes[0, 1].legend()
			axes[0, 1].set_xticks(range(len(first_model_commits)))
			axes[0, 1].set_xticklabels(first_model_commits, rotation=45, ha="right")
		else:
			axes[0, 1].text(
				0.5,
				0.5,
				"No token throughput data",
				ha="center",
				va="center",
				transform=axes[0, 1].transAxes,
			)
			axes[0, 1].set_title("Throughput (Tokens) - N/A")

		# Plot 3: Iteration Time
		for idx, (model_name, metrics_list) in enumerate(models_data.items()):
			commits = [m["commit_hash"] for m in metrics_list]
			iter_times = [m["avg_iter_time"] for m in metrics_list]
			color = colors[idx % len(colors)]
			axes[1, 0].plot(
				range(len(commits)), iter_times, marker="o", linewidth=2, color=color, label=model_name
			)
		axes[1, 0].set_xlabel("Commits")
		axes[1, 0].set_ylabel("Time (seconds)")
		axes[1, 0].set_title("Average Iteration Time")
		axes[1, 0].grid(True, alpha=0.3)
		axes[1, 0].legend()
		axes[1, 0].set_xticks(range(len(first_model_commits)))
		axes[1, 0].set_xticklabels(first_model_commits, rotation=45, ha="right")

		# Plot 4: Peak Memory
		for idx, (model_name, metrics_list) in enumerate(models_data.items()):
			commits = [m["commit_hash"] for m in metrics_list]
			max_memory = [m["max_memory"] for m in metrics_list]
			color = colors[idx % len(colors)]
			axes[1, 1].plot(
				range(len(commits)), max_memory, marker="o", linewidth=2, color=color, label=model_name
			)
		axes[1, 1].set_xlabel("Commits")
		axes[1, 1].set_ylabel("Memory (GB)")
		axes[1, 1].set_title("Peak Memory Usage")
		axes[1, 1].grid(True, alpha=0.3)
		axes[1, 1].legend()
		axes[1, 1].set_xticks(range(len(first_model_commits)))
		axes[1, 1].set_xticklabels(first_model_commits, rotation=45, ha="right")

		plt.tight_layout()

		output_file = output_dir / f"evolution_{gpu_count}gpu.png"
		plt.savefig(output_file, dpi=150, bbox_inches="tight")
		plt.close()

		print(f"Saved plot: {output_file}")


def detect_regressions(
	metrics_by_config: Dict[str, List[Tuple]], threshold_pct: float = 5.0
) -> List[Dict]:
	"""
	Detect performance regressions between consecutive commits.

	Args:
		metrics_by_config: Metrics organized by configuration
		threshold_pct: Percentage threshold for regression detection (default: 5%)

	Returns:
		List of regression events
	"""
	regressions = []

	for config_key, metrics_list in metrics_by_config.items():
		if len(metrics_list) < 2:
			continue

		for i in range(1, len(metrics_list)):
			prev = metrics_list[i - 1]
			curr = metrics_list[i]

			has_regression = False
			regression_details = []

			if prev["mfu"] is not None and curr["mfu"] is not None:
				mfu_change = (curr["mfu"] - prev["mfu"]) / prev["mfu"] * 100
				if mfu_change < -threshold_pct:
					has_regression = True
					regression_details.append(
						f"MFU decreased by {-mfu_change:.1f}% "
						f"({prev['mfu'] * 100:.2f}% → {curr['mfu'] * 100:.2f}%)"
					)
			else:
				throughput_change = (
					(curr["throughput_samples"] - prev["throughput_samples"])
					/ prev["throughput_samples"]
					* 100
				)
				if throughput_change < -threshold_pct:
					has_regression = True
					regression_details.append(
						f"Throughput decreased by {-throughput_change:.1f}% "
						f"({prev['throughput_samples']:.1f} → {curr['throughput_samples']:.1f} samples/s)"
					)

			memory_change = (curr["max_memory"] - prev["max_memory"]) / prev["max_memory"] * 100

			iter_time_change = (
				(curr["avg_iter_time"] - prev["avg_iter_time"]) / prev["avg_iter_time"] * 100
			)

			if memory_change > threshold_pct:
				has_regression = True
				regression_details.append(
					f"Memory increased by {memory_change:.1f}% "
					f"({prev['max_memory']:.2f} → {curr['max_memory']:.2f} GB)"
				)

			if iter_time_change > threshold_pct:
				has_regression = True
				regression_details.append(
					f"Iteration time increased by {iter_time_change:.1f}% "
					f"({prev['avg_iter_time']:.3f} → {curr['avg_iter_time']:.3f} s)"
				)

			if has_regression:
				regressions.append(
					{
						"config": config_key,
						"prev_commit": prev["commit_hash"],
						"curr_commit": curr["commit_hash"],
						"commit_date": curr["commit_date"],
						"commit_message": curr["commit_message"],
						"details": regression_details,
					}
				)

	return regressions


def print_regression_report(regressions: List[Dict]):
	"""Print a formatted regression report."""
	if not regressions:
		print("\n✅ No performance regressions detected!")
		return

	print(f"\n⚠️  Detected {len(regressions)} potential regression(s):\n")
	print("=" * 80)

	for i, reg in enumerate(regressions, 1):
		print(f"\nRegression #{i}: {reg['config']}")
		print(f"  Commit: {reg['curr_commit']} ({reg['commit_date']})")
		print(f"  Message: {reg['commit_message']}")
		print(f"  Previous: {reg['prev_commit']}")
		print("  Issues:")
		for detail in reg["details"]:
			print(f"    • {detail}")

	print("\n" + "=" * 80)


def main():
	parser = argparse.ArgumentParser(
		description="Analyze performance evolution and detect regressions across git commits"
	)
	parser.add_argument(
		"--history-dir",
		type=str,
		default="results/history",
		help="Directory containing benchmark results with metadata",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="results/evolution_plots",
		help="Directory to save visualization plots",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=5.0,
		help="Regression detection threshold in percent (default: 5.0)",
	)
	parser.add_argument(
		"--no-plots", action="store_true", help="Skip generating plots (only detect regressions)"
	)

	args = parser.parse_args()

	history_dir = Path(args.history_dir)
	output_dir = Path(args.output_dir)

	history = load_benchmark_history(history_dir)

	if not history:
		print("No benchmark history found. Run benchmarks with metadata collection first.")
		return

	print(f"\nAnalyzing performance across {len(history)} benchmark runs...")

	metrics_by_config = extract_metrics_by_config(history)

	print(f"\nFound {len(metrics_by_config)} different configurations:")
	for config_key in metrics_by_config:
		print(f"  • {config_key}: {len(metrics_by_config[config_key])} data points")

	regressions = detect_regressions(metrics_by_config, threshold_pct=args.threshold)
	print_regression_report(regressions)

	if not args.no_plots:
		print(f"\nGenerating evolution plots in {output_dir}...")
		plot_evolution(metrics_by_config, output_dir)
		print(f"\n✅ Analysis complete! Plots saved to {output_dir}")
	else:
		print("\n✅ Regression analysis complete!")


if __name__ == "__main__":
	main()
