import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files into DataFrames and plot each on a subplot
files = ['afab.csv', '1f1b.csv', 'i1f1b.csv', 'hanayo-1w.csv', 'hanayo-2w.csv', 'hanayo-3w.csv']
prefix = "results/GPTHanayo"
files = [f'{prefix}/{f}' for f in files]
labels = ["GPipe", "1f1b", "Megatron-LM", "Hanayo 1-Wave", "Hanayo 2-Waves", "Hanayo 3-Waves"]
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()

for idx, file in enumerate(files):
    df = pd.read_csv(f'{file}')

    # Select columns by index (2 to 6)
    mbs = df.iloc[:, 0]
    idle_ratios = df.iloc[:, 2:6]
    total_times = df.iloc[:, 1]
    idle_times = np.zeros(idle_ratios.shape)
    for i, t in enumerate(total_times):
        idle_times[i, :] = idle_ratios.iloc[i, :] * t

    bar_width = 0.1
    xlog = np.log2(mbs)
    bar_positions = np.arange(len(xlog))

    # Create a bar plot for the selected columns
    for d in range(idle_times.shape[1]):
        pos = bar_positions + d * (bar_width * 1.1) - ((idle_times.shape[1] - 1) * bar_width * 1.1) / 2
        i = axs[idx].bar(pos, idle_times[:, d], width=bar_width, align="center", color="tab:gray")
        if d == 0: i.set_label('Idle')
        axs[idx].bar(pos, total_times[:] - idle_times[:, d], bottom=idle_times[:, d], width=bar_width, align="center", label=f'Device {d}')

    axs[idx].legend()
    axs[idx].set_xlabel('Micro batches size with fixed batch size = 64', fontdict={'size': 14})
    axs[idx].set_xticks(ticks=xlog)
    axs[idx].set_xticklabels(mbs)
    axs[idx].set_ylabel('Median time for one iteration (s)', fontdict={'size': 14})
    axs[idx].set_title(f'{labels[idx]}')

plt.tight_layout()
plt.show()

