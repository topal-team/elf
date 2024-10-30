import matplotlib.pyplot as plt
import os

# Read data from files
files = ["1f1b.csv", "i1f1b.csv", "afab.csv", "hanayo-1w.csv", "hanayo-2w.csv"]
prefix = "GPTHanayo"
files = [f"results/{prefix}/{f}" for f in files]
labels = ["1f1b", "Megatron", "GPipe", "Hanayo 1-Wave", "Hanayo 2-Waves"]
labels = [label for file, label in zip(files, labels) if os.path.exists(file)]

# files = ["tmp1f1b.csv", "tmpmegatron.csv"]
# labels = ["1f1b", "Megatron"]

data = []
for i, f in enumerate(files):
	if not os.path.exists(f):
		continue
	with open(f, "r") as f:
		f.readline()  # skip header line
		data.append([line.strip().split(",") for line in f])

# Extract sizes and times from the data
st = [tuple(zip(*d)) for d in data]
sizes = map(lambda st: st[0], st)
times = map(lambda st: st[1], st)
memories = map(lambda st: st[6:], st)

# Convert strings to floats
sizes = [list(map(float, s)) for s in sizes]
times = [list(map(float, t)) for t in times]
memories = [[list(map(float, d)) for d in m] for m in memories]


fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 3)

# Merge the first row into one subplot
ax1 = fig.add_subplot(gs[0, 0:2])

# Plot the curves
for i, (s, t) in enumerate(zip(sizes, times)):
	ax1.plot(s, t, label=labels[i])

ax1.set_ylabel("Median time for one iteration (s)", fontdict={"size": 14})
ax1.set_xlabel("Micro batches size", fontdict={"size": 14})
ax1.tick_params(axis="both", which="major", labelsize=14)
ax1.set_xscale("log", base=2)
ax1.set_ylim(0)
ax1.legend(prop={"size": 12})
ax1.set_title("Time", fontdict={"size": 16})

# Plot the remaining subplots
axes = []
for i, j in [(0, 2), (1, 0), (1, 1), (1, 2)]:
	ax = fig.add_subplot(gs[i, j])
	axes.append(ax)

for i, (s, mem) in enumerate(zip(sizes, memories)):
	for d, m in enumerate(mem):
		axes[d].plot(s, m, label=labels[i])

for i, ax in enumerate(axes):
	ax.set_xlabel("Micro batches size with fixed batch size = 64", fontdict={"size": 14})
	ax.tick_params(axis="both", which="major", labelsize=14)
	ax.set_xscale("log", base=2)
	ax.set_ylim(0)
	ax.legend(prop={"size": 10})
	ax.set_ylabel("Peak memory used (GB)", fontdict={"size": 14})
	ax.set_title(f"Memory on device {i}", fontdict={"size": 16})

fig.suptitle("Comparison of pipelining schedules", fontsize=20)
plt.tight_layout()
plt.show()
