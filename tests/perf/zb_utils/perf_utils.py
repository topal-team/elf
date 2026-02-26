import time
import numpy as np
import torch
from scipy import stats


def benchmark_operation(operation, n_trials=50, n_iters_per_trial=200, n_warmup=100):
	"""
	Benchmark a PyTorch operation with robust statistics.

	Args:
		operation: Callable that performs the operation to benchmark
		n_trials: Number of independent trials
		n_iters_per_trial: Number of iterations per trial
		n_warmup: Number of warmup iterations

	Returns:
		List of timing measurements in milliseconds
	"""
	# Warmup to stabilize GPU frequency and kernel compilation
	for _ in range(n_warmup):
		operation()

	if torch.cuda.is_available():
		torch.cuda.synchronize()

	# Benchmark - multiple trials
	times = []
	for _ in range(n_trials):
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		for _ in range(n_iters_per_trial):
			operation()
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		t1 = time.perf_counter()
		times.append((t1 - t0) * 1000)

	return times


def compute_statistics(times):
	"""
	Compute robust statistics from timing measurements.

	Args:
		times: List or array of timing measurements

	Returns:
		Dictionary containing mean, median, std, CV, min, max
	"""
	times_arr = np.array(times)
	mean = np.mean(times_arr)
	median = np.median(times_arr)
	std = np.std(times_arr, ddof=1)
	cv = std / mean * 100

	return {
		"mean": mean,
		"median": median,
		"std": std,
		"cv": cv,
		"min": np.min(times_arr),
		"max": np.max(times_arr),
		"times": times_arr,
	}


def print_performance_report(
	name, stats_baseline, stats_test, device, n_trials, n_iters_per_trial, n_warmup, **config_params
):
	"""
	Print a formatted performance comparison report.

	Args:
		name: Name of the operation being tested (e.g., "Conv1d", "Linear")
		stats_baseline: Statistics dict from baseline implementation
		stats_test: Statistics dict from test implementation
		device: torch.device being used
		n_trials: Number of trials
		n_iters_per_trial: Iterations per trial
		n_warmup: Number of warmup iterations
		**config_params: Additional configuration parameters to display
	"""
	overhead_mean = (stats_test["mean"] - stats_baseline["mean"]) / stats_baseline["mean"] * 100
	overhead_median = (
		(stats_test["median"] - stats_baseline["median"]) / stats_baseline["median"] * 100
	)

	# Perform paired t-test
	t_stat, p_value = stats.ttest_rel(stats_test["times"], stats_baseline["times"])

	# Compute 95% confidence interval for the difference
	diff = stats_test["times"] - stats_baseline["times"]
	diff_mean = np.mean(diff)
	diff_std = np.std(diff, ddof=1)
	ci_95 = 1.96 * diff_std / np.sqrt(n_trials)

	print("\n" + "=" * 70)
	print(f"{name} Performance Analysis")
	print("=" * 70)
	print(f"Device: {device}")
	print(f"Config: {n_trials} trials × {n_iters_per_trial} iterations (warmup: {n_warmup})")

	if config_params:
		config_str = ", ".join(f"{k}={v}" for k, v in config_params.items())
		print(f"        {config_str}")

	print(f"\nRegular {name}:")
	print(
		f"  Mean:   {stats_baseline['mean']:.2f}ms (±{stats_baseline['std']:.2f}ms, CV={stats_baseline['cv']:.1f}%)"
	)
	print(f"  Median: {stats_baseline['median']:.2f}ms")
	print(f"  Range:  [{stats_baseline['min']:.2f}, {stats_baseline['max']:.2f}]ms")

	print(f"\nDecoupled {name}DW:")
	print(
		f"  Mean:   {stats_test['mean']:.2f}ms (±{stats_test['std']:.2f}ms, CV={stats_test['cv']:.1f}%)"
	)
	print(f"  Median: {stats_test['median']:.2f}ms")
	print(f"  Range:  [{stats_test['min']:.2f}, {stats_test['max']:.2f}]ms")

	print("\nOverhead Analysis:")
	print(f"  Mean overhead:   {overhead_mean:.1f}%")
	print(f"  Median overhead: {overhead_median:.1f}%")
	print(f"  Absolute diff:   {diff_mean:.2f}ms ± {ci_95:.2f}ms (95% CI)")

	print("\nStatistical Test (paired t-test):")
	print(f"  t-statistic: {t_stat:.3f}")
	print(f"  p-value:     {p_value:.4f}")
	print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
	print("=" * 70)

	return overhead_mean, overhead_median


def assert_performance(stats_baseline, stats_test, device, max_overhead_pct=25, max_cv_pct=None):
	"""
	Assert that performance meets quality thresholds.

	Args:
		stats_baseline: Statistics dict from baseline
		stats_test: Statistics dict from test implementation
		device: torch.device being used
		max_overhead_pct: Maximum allowed median overhead percentage
		max_cv_pct: Maximum allowed CV percentage (defaults: GPU=10%, CPU=15%)
	"""
	if max_cv_pct is None:
		max_cv_pct = 10 if device.type == "cuda" else 15

	overhead_median = (
		(stats_test["median"] - stats_baseline["median"]) / stats_baseline["median"] * 100
	)

	assert overhead_median < max_overhead_pct, (
		f"Decoupled implementation median overhead is {overhead_median:.1f}%, should be < {max_overhead_pct}%"
	)
	assert stats_baseline["cv"] < max_cv_pct, (
		f"Baseline measurements too variable (CV={stats_baseline['cv']:.1f}%), check system load"
	)
	assert stats_test["cv"] < max_cv_pct, (
		f"Test implementation measurements too variable (CV={stats_test['cv']:.1f}%), check system load"
	)
