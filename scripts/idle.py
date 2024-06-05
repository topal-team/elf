import matplotlib.pyplot as plt
import os
import numpy as np

files = ['afab.csv', '1f1b.csv', 'i1f1b.csv', 'hanayo-1w.csv', 'hanayo-2w.csv', 'hanayo-3w.csv']
prefix = "GPTHanayo"
files = [f'results/{prefix}/{f}' for f in files]
labels = ["GPipe", "1f1b", "Megatron-LM", "Hanayo 1-Wave", "Hanayo 2-Waves", "Hanayo 3-Waves"]
labels = [l for f,l in zip(files, labels) if os.path.exists(f)]


data = []
for i, f in enumerate(files):
    if not os.path.exists(f):
        continue
    with open(f, "r") as f:
        f.readline() # skip header line
        data.append([line.strip().split(',') for line in f])

# Extract sizes and times from the data
st = [tuple(zip(*d)) for d in data]
sizes = map(lambda st: st[0], st)
times = map(lambda st: st[1], st)
idles = map(lambda st: st[2:6], st)

# Convert strings to floats
sizes = [list(map(float, s)) for s in sizes]
times = [np.array(list(map(float, t))) for t in times]
idles = [[np.array(list(map(float, d))) for d in t] for t in idles]

bar_width = 0.1

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
for k in range(4):
    ax = axs[k // 2, k % 2]
    xlog = [np.log2(s) for s in sizes]
    bar_positions = np.arange(len(xlog[0]))

    for j in range(len(sizes)):
        # times[j] = np.pad(times[j], (0, len(xlog[0]) - len(times[j])), constant_values = 0)
        # idles[j][k] = np.pad(idles[j][k], (0, len(xlog[0]) - len(idles[j][k])), constant_values = 0)

        idle_time = idles[j][k] * times[j]

        i = ax.bar(bar_positions + j * (bar_width * 1.1), idle_time, color = 'tab:gray', width=bar_width, align='center')
        if j == 0: i.set_label('Idle')
        ax.bar(bar_positions + j * (bar_width * 1.1), times[j] - idle_time, width=bar_width, bottom=idle_time, align='center', label = 'Compute')

    ax.set_ylabel('Median time for one iteration (s)', fontdict={'size': 14})
    ax.set_xlabel('Micro batches size with fixed batch size = 64', fontdict={'size': 14})
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xticks(bar_positions + (bar_width * 1.1) * (len(sizes) - 1) / 2)
    ax.set_xticklabels(sizes[0])  # Assuming all sizes are the same for x-axis labels
    ax.set_ylim(0)

    # Indicate which bar corresponds to which label
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ['Idle time'] + labels, prop={'size': 12}, loc='upper center')
    ax.set_title(f'Device {k}')

plt.suptitle('Idle & Compute time')
plt.tight_layout()
plt.show()