import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality plots
print(plt.style.available)
plt.style.use("seaborn-v0_8-dark-palette")
plt.rcParams.update(
	{
		"font.size": 12,
		"axes.labelsize": 14,
		"axes.titlesize": 16,
		"xtick.labelsize": 12,
		"ytick.labelsize": 12,
		"legend.fontsize": 12,
		"figure.figsize": (12, 6),  # Made figure taller
		"figure.dpi": 300,
	}
)

# Load and process data
df = pd.read_csv("results.csv")
names = df["name"].unique()

# Create figure with two subplots and extra bottom space for legend
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=False)
plt.subplots_adjust(bottom=0.2)  # Added more space at bottom

# Color palette for bars
colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

# Time subplot
bar_width = 0.15
x = np.arange(1)

for i, (name, color) in enumerate(zip(names, colors)):
	time = df[df["name"] == name]["total_time_0"].iloc[0]
	ax1.bar(
		x + i * bar_width,
		time,
		width=bar_width,
		label=name,
		color=color,
		edgecolor="black",
		linewidth=1,
	)

ax1.set_xticks([])
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylabel("Execution Time (s)")
ax1.set_title("(a) Iteration Time")
ax1.grid(axis="y", linestyle="--", alpha=0.7)

# Memory subplot
x = np.arange(4)
x_labels = [f"GPU {d}" for d in range(4)]

for i, (name, color) in enumerate(zip(names, colors)):
	mems = [df[df["name"] == name][f"mem_{d}"].iloc[0] for d in range(4)]
	print(f"{name} - {mems}")
	ax2.bar(
		x + i * bar_width,
		mems,
		width=bar_width,
		label=name,
		color=color,
		edgecolor="black",
		linewidth=1,
	)

ax2.set_xticks(x + bar_width * (len(names) - 1) / 2)
ax2.set_xticklabels(x_labels)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_ylabel("Peak Memory Usage (GB)")
ax2.set_title("(b) Memory Footprint")
ax2.grid(axis="y", linestyle="--", alpha=0.7)

# Add single legend below both subplots
handles, labels = ax2.get_legend_handles_labels()
fig.legend(
	handles,
	labels,
	loc="center",
	bbox_to_anchor=(0.5, 0.05),  # Moved legend down more
	ncol=len(names),
	frameon=False,
)

plt.show()
