import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mode = sys.argv[1] if len(sys.argv) > 1 else "device"
# Load the CSV files into DataFrames and plot each on a subplot
df_full = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "results/GPTHanayo/full.csv")

if mode == "device":
	fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
	axs = axs.flatten()

	# Every schedule
	for idx, name in enumerate(df_full["name"].unique()):
		df = df_full[df_full["name"] == name]

		mbs = df["mb_size"]
		total_times = df.filter(regex="^total_time")
		start_times = df.filter(regex="^start_time")
		end_times = df.filter(regex="^end_time")
		bubble_times = df.filter(regex="^bubble_time")

		bar_width = 0.1
		xlog = np.log2(mbs)
		bar_positions = np.arange(len(xlog))

		# Create a bar plot for the selected columns
		for d in range(total_times.shape[1]):
			pos = (
				bar_positions + d * (bar_width * 1.2) - ((total_times.shape[1] - 1) * bar_width * 1.2) / 2
			)
			bottom = np.zeros(total_times.shape[0])

			i = axs[idx].bar(
				pos, start_times.iloc[:, d], width=bar_width, align="center", color="dodgerblue"
			)
			bottom += start_times.iloc[:, d]
			if d == 0:
				i.set_label("Start idle")

			i = axs[idx].bar(
				pos,
				bubble_times.iloc[:, d],
				width=bar_width,
				bottom=bottom,
				align="center",
				color="hotpink",
			)
			bottom += bubble_times.iloc[:, d]
			if d == 0:
				i.set_label("Bubble")

			i = axs[idx].bar(
				pos, end_times.iloc[:, d], width=bar_width, bottom=bottom, align="center", color="orange"
			)
			bottom += end_times.iloc[:, d]
			if d == 0:
				i.set_label("End idle")

			axs[idx].bar(
				pos,
				total_times.iloc[:, 0] - bottom,
				bottom=bottom,
				width=bar_width,
				align="center",
				label=f"Device {d}",
			)

		axs[idx].legend()
		axs[idx].set_xlabel("Micro batches size with fixed batch size = 64", fontdict={"size": 14})
		axs[idx].set_xticks(ticks=xlog)
		axs[idx].set_xticklabels(mbs)
		axs[idx].set_ylabel("Median time for one iteration (s)", fontdict={"size": 14})
		axs[idx].set_title(name)

elif mode == "schedule":
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	axs = axs.flatten()

	for d in range(4):  # hardcoded number of gpus :)
		mbs = df_full["mb_size"].unique()
		n_schedules = df_full["name"].unique().shape[0]

		bar_width = 0.1
		xlog = np.log2(mbs)
		bar_positions = np.arange(len(xlog))

		# Create a bar plot for the selected columns
		for idx, name in enumerate(df_full["name"].unique()):
			total_times = df_full[df_full["name"] == name][f"total_time_{d}"]
			start_times = df_full[df_full["name"] == name][f"start_time_{d}"]
			bubble_times = df_full[df_full["name"] == name][f"bubble_time_{d}"]
			end_times = df_full[df_full["name"] == name][f"end_time_{d}"]

			pos = bar_positions + (bar_width * 1.2) * idx - (bar_width * 1.2 * (n_schedules - 1) / 2)
			bottom = np.zeros(total_times.shape[0])

			i = axs[d].bar(pos, start_times, width=bar_width, align="center", color="dodgerblue")
			bottom += start_times
			if idx == 0:
				i.set_label("Start idle")

			i = axs[d].bar(
				pos, bubble_times, width=bar_width, bottom=bottom, align="center", color="hotpink"
			)
			bottom += bubble_times
			if idx == 0:
				i.set_label("Bubble")

			i = axs[d].bar(pos, end_times, width=bar_width, bottom=bottom, align="center", color="orange")
			bottom += end_times
			if idx == 0:
				i.set_label("End idle")

			axs[d].bar(
				pos, total_times - bottom, bottom=bottom, width=bar_width, align="center", label=name
			)

		axs[d].legend()
		axs[d].set_xlabel("Micro batches size with fixed batch size = 64", fontdict={"size": 14})
		axs[d].set_xticks(ticks=xlog)
		axs[d].set_xticklabels(mbs)
		axs[d].set_ylabel("Median time for one iteration (s)", fontdict={"size": 14})
		axs[d].set_title(f"Device {d}")

elif mode == "mems":
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	axs = axs.flatten()

	for d in range(4):  # hardcoded number of gpus :)
		mbs = df_full["mb_size"].unique()
		n_schedules = df_full["name"].unique().shape[0]
		for idx, name in enumerate(df_full["name"].unique()):
			peak_mem = df_full[df_full["name"] == name][f"mem_{d}"]
			axs[d].plot(mbs, peak_mem, label=name)

		axs[d].set_xlabel("Micro batches size with fixed batch size = 64", fontdict={"size": 14})
		axs[d].tick_params(axis="both", which="major", labelsize=14)
		axs[d].set_xscale("log", base=2)
		axs[d].set_ylim(0)
		axs[d].legend(prop={"size": 10})
		axs[d].set_ylabel("Peak memory used (GB)", fontdict={"size": 14})
		axs[d].set_title(f"Memory on device {d}", fontdict={"size": 16})

elif mode == "remat":
	df_r = pd.read_csv("results/GPTHanayo/remat/full.csv")

	fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
	axs[0, 0].remove()  # Remove the second subplot in the first row
	axs[0, 1].remove()  # Remove the second subplot in the first row
	axs[0, 0] = fig.add_subplot(2, 3, (1, 2))  # Create a new subplot spanning the first two columns
	axs = axs.flatten()

	for d in range(4):  # Assuming 4 devices as in previous sections
		mbs = df_full["mb_size"].unique()
		filtered_names = df_full["name"].unique()
		filtered_names = [
			name
			for name in filtered_names
			if name not in ["GPipe", "1f1b", "Hanayo 2-Waves", "Hanayo 3-Waves"]
		]
		n_schedules = len(filtered_names)

		for idx, name in enumerate(filtered_names):
			if d == 0:
				total_time_full = df_full[df_full["name"] == name][f"total_time_{d}"]
				total_time_r = df_r[df_r["name"] == name][f"total_time_{d}"]
			mem_full = df_full[df_full["name"] == name][f"mem_{d}"]
			mem_r = df_r[df_r["name"] == name][f"mem_{d}"]

			if d == 0:
				axs[d].plot(mbs, total_time_full, label=name)
				axs[d].plot(mbs, total_time_r, label=f"{name} w/ remat")
			axs[d + 2].plot(mbs, mem_full, label=name)
			axs[d + 2].plot(mbs, mem_r, label=f"{name} w/ remat")

		if d == 0:
			axs[d].set_xlabel("Micro batches size with fixed batch size = 64", fontdict={"size": 14})
			axs[d].tick_params(axis="both", which="major", labelsize=14)
			axs[d].set_xscale("log", base=2)
			axs[d].set_ylabel("Iteration time (s)", fontdict={"size": 14})
			axs[d].set_title("Time", fontdict={"size": 16})
			axs[d].legend()

		axs[d + 2].set_xlabel("Micro batches size with fixed batch size = 64", fontdict={"size": 14})
		axs[d + 2].tick_params(axis="both", which="major", labelsize=14)
		axs[d + 2].set_xscale("log", base=2)
		axs[d + 2].set_ylabel("Peak memory usage (GB)", fontdict={"size": 14})
		axs[d + 2].set_title(f"Memory on device {d}", fontdict={"size": 16})
		axs[d + 2].legend()

elif mode == "partitioner":
	del df_full
	fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
	axs = axs.flatten()

	partitioners = ["default", "metis", "dagP", "constrained", "old"]
	device = 3

	dfs = [pd.read_csv(f"results/GPTLarge/{p}.csv") for p in partitioners]

	mbs = dfs[0]["mb_size"].unique()
	filtered_names = dfs[0]["name"].unique()
	filtered_names = [
		name
		for name in filtered_names
		if name in ["GPipe", "1f1b", "Megatron-LM", "Hanayo 1-Wave", "Hanayo 2-Waves", "Hanayo 3-Waves"]
	]
	n_schedules = len(filtered_names)

	for idx, name in enumerate(filtered_names):
		bar_width = 0.15
		x = np.arange(len(mbs))
		for j, (df, p) in enumerate(zip(dfs, partitioners)):
			empty = df[df["name"] == name].empty
			start_idle = (
				df[df["name"] == name][f"start_time_{device}"] if not empty else np.zeros((len(mbs),))
			)
			bubble = (
				df[df["name"] == name][f"bubble_time_{device}"] if not empty else np.zeros((len(mbs),))
			)
			end_idle = (
				df[df["name"] == name][f"end_time_{device}"] if not empty else np.zeros((len(mbs),))
			)
			total = df[df["name"] == name][f"total_time_{device}"] if not empty else np.zeros((len(mbs),))

			bottom = 0

			i = axs[idx].bar(
				x + j * bar_width, start_idle, width=bar_width, bottom=bottom, color="dodgerblue"
			)
			if j == 0:
				i.set_label("Start idle")
			bottom += start_idle

			i = axs[idx].bar(x + j * bar_width, bubble, width=bar_width, bottom=bottom, color="hotpink")
			bottom += bubble
			if j == 0:
				i.set_label("Bubble")

			i = axs[idx].bar(x + j * bar_width, end_idle, width=bar_width, bottom=bottom, color="orange")
			bottom += end_idle
			if j == 0:
				i.set_label("End idle")

			axs[idx].bar(x + j * bar_width, total, width=bar_width, bottom=bottom, label=f"{p}")

		axs[idx].set_xlabel("Micro batches size with fixed batch size = 64", fontdict={"size": 14})
		axs[idx].set_xticks(x + bar_width * (len(partitioners) - 1) / 2)
		axs[idx].set_xticklabels(mbs)
		axs[idx].tick_params(axis="both", which="major", labelsize=14)
		axs[idx].set_ylabel("Idle time (s)", fontdict={"size": 14})
		axs[idx].set_title(name, fontdict={"size": 16})
		axs[idx].legend()

plt.tight_layout()
plt.show()
