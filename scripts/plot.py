import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# --- Load data ---

# files = ['afab.csv', '1f1b.csv', 'i1f1b.csv', 'hanayo.csv']
# labels = ["GPipe", "1f1b", "Megatron-LM", "Hanayo 1-Wave"]

files = ['i1f1b.csv']
labels = ["Megatron"]

prefix = 'results/GPTHanayo' 
dfs = []
for f in files:
    dfs.append(pd.read_csv(f'{prefix}/{f}'))

prefix = 'results/GPTHanayo/remat'
dfs_remat = []
for f in files:
    dfs_remat.append(pd.read_csv(f'{prefix}/{f}'))

prefix = 'results/GPTHanayo/offload'
dfs_off = []
for f in files:
    dfs_off.append(pd.read_csv(f'{prefix}/{f}'))

mode = sys.argv[1] if len(sys.argv) > 1 else "mems"

if mode == "mems":
    # --- Create layout ---

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 3)

    # Merge the first row into one subplot
    ax1 = fig.add_subplot(gs[0, 0:2])

    # Plot the remaining subplots
    axes = []
    for i, j in [(0, 2), (1, 0), (1, 1), (1, 2)]:
        ax = fig.add_subplot(gs[i, j])
        axes.append(ax)

    # --- Plot data ---

    for df, df_r, df_o, label in zip(dfs, dfs_remat, dfs_off, labels):
        ax1.plot(df['mb_size'], df['total_time'], label = label)
        ax1.plot(df_r['mb_size'], df_r['total_time'], label = f'{label} w/ remat')
        ax1.plot(df_o['mb_size'], df_o['total_time'], label = f'{label} w/ offloading')

    ax1.set_xscale('log', base = 2)
    ax1.legend()
    ax1.set_xlabel('Micro batch size, full batch size = 64')
    ax1.set_ylabel('Iteration time (s)')
    ax1.set_title('Time comparison')

    for i, ax in enumerate(axes):
        for df, df_r, df_o, label in zip(dfs, dfs_remat, dfs_off, labels):
            ax.plot(df['mb_size'], df[f'mem_{i}'], label = label)
            ax.plot(df_r['mb_size'], df_r[f'mem_{i}'], label = f'{label} w/ remat')
            ax.plot(df_o['mb_size'], df_o[f'mem_{i}'], label = f'{label} w/ offloading')
            ax.set_xscale('log', base = 2)
            ax.legend()
            ax.set_xlabel('Micro batch size')
            ax.set_ylabel('Peak memory used (GB)')
            ax.set_title(f'Memory on GPU {i}')

elif mode == "idles":
    # --- Layout ---
    
    fig, axes = plt.subplots(2, 2)
    axes = [ax for row in axes for ax in row]
    bar_width = 0.1
    space = 1.1

    # --- Data ---

    for i, ax in enumerate(axes):
        for df, df_r, df_o, label in zip(dfs, dfs_remat, dfs_off, labels):
            xlog = np.log2(df['mb_size'])
            bar_positions = np.arange(len(xlog))

            idle_times = df[f'idle_ratio_{i}'] * df['total_time']
            compute_times = df['total_time']
            ax.bar(bar_positions, idle_times, color = "tab:gray", width = bar_width, label = 'Idle')
            ax.bar(bar_positions, compute_times, bottom = idle_times, width = bar_width, label = label)

            idle_times = df_r[f'idle_ratio_{i}'] * df_r['total_time']
            compute_times = df_r['total_time'] - idle_times
            ax.bar(bar_positions + space * bar_width, idle_times, color = "tab:gray", width = bar_width)
            ax.bar(bar_positions + space * bar_width, compute_times, bottom = idle_times, width = bar_width, label = f'{label} w/ remat')

            idle_times = df_o[f'idle_ratio_{i}'] * df_o['total_time']
            compute_times = df_o['total_time'] - idle_times
            ax.bar(bar_positions + 2 * space * bar_width, idle_times, color = "tab:gray", width = bar_width)
            ax.bar(bar_positions + 2 * space * bar_width, compute_times, bottom = idle_times, width = bar_width, label = f'{label} w/ offload')

            ax.set_xticks(bar_positions + (bar_width * 1.1 * (3 - 1)) / 2)
            ax.set_xticklabels(df['mb_size'])
            ax.legend()
            ax.set_xlabel('Micro batch size')
            ax.set_ylabel('Iteration time (s)')
            ax.set_title(f'Idle time on GPU {i}')

plt.suptitle('GPT-like model training with batch size = 64', fontsize=20)
fig.tight_layout()
plt.show()