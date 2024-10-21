import torch
import torch.nn as nn
import torch.distributed as dist
import os
import sys
import json
import time

# Add the parent directory to the Python path
sys.path.append("./")

from pipeline import Pipeline
from pipeline.utils import Timer, pretty_print_params
from models.simple import SimpleTransformer


def run_experiment(num_epochs):
	# Initialize distributed environment
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	torch.cuda.set_device(local_rank)

	# Create a large SimpleTransformer model
	input_dim = 50000
	hidden_dim = 1024
	n_blocks = 12
	model = SimpleTransformer(input_dim, hidden_dim, n_blocks)
	if rank == 0:
		print(
			"# of trainable parameters : ",
			pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
		)

	# Prepare sample input
	batch_size = 32
	seq_length = 128
	sample = torch.randint(0, input_dim, (batch_size, seq_length)).cuda()

	# Wrap the model in Pipeline
	pipe_creation_start = time.time()
	model = Pipeline(model, sample, schedule="1f1b")
	pipe_creation_time = time.time() - pipe_creation_start

	# Define loss function and optimizer
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	# Training loop
	times = []
	for epoch in range(num_epochs):
		if rank == 0:
			print(f"Epoch {epoch + 1}/{num_epochs}")

		# Generate random input and target for this epoch
		input_data = torch.randint(0, input_dim, (batch_size, seq_length)).cuda()
		target = torch.randint(0, input_dim, (batch_size, seq_length)).cuda()

		with Timer() as timer:
			# Forward pass
			optimizer.zero_grad()
			_ = model(input_data, target, loss_fn, split_size=batch_size // dist.get_world_size())
			optimizer.step()

		measured = timer.time()
		internal_time = model.times["total"]

		times.append({"epoch": epoch + 1, "measured_time": measured, "internal_time": internal_time})

		if rank == 0:
			print(f"Iteration execution time: {measured:.4f} seconds")
			print(f"Internal capture : {internal_time:.4f} seconds")

	return times, pipe_creation_time


def main():
	num_runs = 5
	num_epochs = 20
	all_results = []
	dist.init_process_group(backend="nccl")

	for run in range(num_runs):
		if int(os.getenv("RANK", "0")) == 0:
			print(f"\nRun {run + 1}/{num_runs}")

		times, pipe_creation_time = run_experiment(num_epochs)
		all_results.append({"run": run + 1, "pipe_creation_time": pipe_creation_time, "epochs": times})

		dist.barrier()

		# Reinitialize for the next run
		if run < num_runs - 1:
			time.sleep(1)  # Wait a bit before reinitializing

	# Write all results to a file
	if int(os.getenv("RANK", "0")) == 0:
		with open("pipeline_times_multiple_runs.json", "w") as f:
			json.dump(all_results, f, indent=4)

	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	import sys
	import json
	import numpy as np

	if len(sys.argv) > 1 and sys.argv[1] == "plot":
		# Load data from JSON files
		with open("results/fx.json", "r") as f:
			fx_data = json.load(f)
		with open("results/export.json", "r") as f:
			export_data = json.load(f)

		# Extract times for all epochs
		fx_times = [[epoch["measured_time"] for epoch in run["epochs"]] for run in fx_data]
		export_times = [[epoch["measured_time"] for epoch in run["epochs"]] for run in export_data]

		# Calculate median over all epochs
		fx_median_all = np.median([time for run in fx_times for time in run])
		export_median_all = np.median([time for run in export_times for time in run])

		# Calculate median for each epoch
		fx_median_per_epoch = np.median(fx_times, axis=0)
		export_median_per_epoch = np.median(export_times, axis=0)

		# Print results
		print(f"FX median time over all epochs: {fx_median_all:.4f} seconds")
		print(f"Export median time over all epochs: {export_median_all:.4f} seconds")

		print("\nMedian times for each epoch:")
		for i, (fx_med, export_med) in enumerate(zip(fx_median_per_epoch, export_median_per_epoch)):
			print(f"Epoch {i+1}: FX: {fx_med:.4f} seconds, Export: {export_med:.4f} seconds")

		# Plot median over epochs using seaborn
		import matplotlib.pyplot as plt
		import seaborn as sns
		import pandas as pd
		from prettytable import PrettyTable

		# Prepare data for seaborn
		epochs = range(1, len(fx_median_per_epoch) + 1)
		data = pd.DataFrame(
			{
				"Epoch": list(epochs) + list(epochs),
				"Time (seconds)": list(fx_median_per_epoch) + list(export_median_per_epoch),
				"Method": ["FX"] * len(epochs) + ["Export"] * len(epochs),
			}
		)

		# Set up the plot style
		sns.set_style("whitegrid")
		sns.set_palette("deep")

		# Create the plot
		plt.figure(figsize=(12, 6))
		ax = sns.lineplot(x="Epoch", y="Time (seconds)", hue="Method", data=data, marker="o")

		# Customize the plot
		plt.title("Median Execution Time per Epoch", fontsize=16)
		plt.xlabel("Epoch", fontsize=12)
		plt.ylabel("Median Time (seconds)", fontsize=12)
		ax.legend(title="Method", title_fontsize="12", fontsize="10")

		# Add value labels
		for line in ax.lines:
			y = line.get_ydata()
			x = line.get_xdata()
			for i, (xi, yi) in enumerate(zip(x, y)):
				ax.annotate(
					f"{yi:.2f}",
					(xi, yi),
					xytext=(0, 5),
					textcoords="offset points",
					ha="center",
					va="bottom",
					fontsize=8,
					alpha=0.7,
				)

		# Adjust layout and save
		plt.tight_layout()
		plt.savefig("fx-export.png", dpi=300)
		plt.close()

		# Compare pipeline creation times
		fx_creation_times = [run["pipe_creation_time"] for run in fx_data]
		export_creation_times = [run["pipe_creation_time"] for run in export_data]

		# Calculate statistics
		fx_mean = np.mean(fx_creation_times)
		fx_median = np.median(fx_creation_times)
		fx_min = np.min(fx_creation_times)
		fx_max = np.max(fx_creation_times)
		fx_stddev = np.std(fx_creation_times)

		export_mean = np.mean(export_creation_times)
		export_median = np.median(export_creation_times)
		export_min = np.min(export_creation_times)
		export_max = np.max(export_creation_times)
		export_stddev = np.std(export_creation_times)

		# Create a table for comparison
		table = PrettyTable()
		table.field_names = ["Metric", "FX", "Export"]
		table.align["Metric"] = "l"
		table.align["FX"] = "r"
		table.align["Export"] = "r"

		table.add_row(["Mean", f"{fx_mean:.2f}s", f"{export_mean:.2f}s"])
		table.add_row(["Median", f"{fx_median:.2f}s", f"{export_median:.2f}s"])
		table.add_row(["Min", f"{fx_min:.2f}s", f"{export_min:.2f}s"])
		table.add_row(["Max", f"{fx_max:.2f}s", f"{export_max:.2f}s"])
		table.add_row(["Std Dev", f"{fx_stddev:.2f}s", f"{export_stddev:.2f}s"])

		# Print comparison
		print("\nPipeline Creation Time Comparison:")
		print(table)

	else:
		main()
