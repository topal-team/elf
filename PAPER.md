# Optimized Forward-Backward Rematerialization for Memory-Efficient Pipeline Parallel Training (NeurIPS 2026)

This branch of ELF contains the scripts and data for the experiments of the paper. The ILP
formulation and the baselines it is compared with are in `pipeline-ilps/`; ELF provides the base
pipeline schedules, the stage profiling and the execution of the rematerialization strategies.

```
pipeline-ilps/   ILP formulation (stage_remat_multiwave.py), Greedy (rarltm_greedy.py), AdaPipe (adapipe.py),
                 solving entry points (runall.py, exp_solving_time.py)
benchmarks/      GPU scripts: profile_stages.py, bench_remat.py (ILP, Greedy, FullRemat, NoRemat),
                 profile_units.py, bench_adapipe.py (AdaPipe); paper_remat.py links them to pipeline-ilps
workflow/        Snakemake workflow running profiling -> solving -> benchmarking on a SLURM cluster
memory-sweep/    Figures 3 and 6: estimated throughput of the ILP solution vs memory budget
solving-time/    Figures 5 and 11: ILP solving time
throughput/      Figures 4, 7, 8, 9, 10 and statistics of Section 4.4: measured throughput on H100 GPUs
```

Each `results/` directory holds the raw data behind the paper's figures. Models `ppcm9-6`, `ppcm21`
and `ppcm38-5` are the 9.6B, 21B and 38.5B Transformers of the paper.

## ELF version

The results of the paper (and the data in the `results/` directories) were obtained with an earlier,
unreleased version of ELF. The code in this branch has been ported to the current version of ELF,
which differs from that version in several ways: rematerialization engine, memory model of the ILP,
causal attention in the benchmark Transformer, and interleaving of the 1F1B-2W schedule. As a
consequence:

- Figures 5, 6 and 11 (ILP solving time, per-microbatch vs uniform rematerialization) use the same
  schedules and ILP formulation as the paper, and are reproduced exactly.
- Figure 3 was computed with an earlier version of the ILP formulation: re-running it gives objective
  values within a few percent of the paper's, and lower values for the 1F1B-2W curve, whose schedule
  changed.
- Figures 4, 7, 8, 9 and 10 are measurements on GPUs with the earlier runtime. New measurements with
  this code will differ from the paper's values.

The code used to produce the results of the paper, with the original version of ELF, is provided as
supplementary material of the paper.

## Requirements

```bash
pip install -e .                     # ELF, from the root of the repository
pip install pulp cplex orloge pandas scipy matplotlib seaborn snakemake
```

The GPU experiments additionally need NVIDIA H100 80GB nodes with 4 GPUs each (up to 4 nodes) and
SLURM. FlashAttention 3 (`flash-attn`, used by the `fatt3` attention backend of the runfiles) is
optional: without it, the flash kernel of PyTorch's SDPA is used.

The experiments use IBM CPLEX through PuLP (`CPLEX_PY`). The pip `cplex` package alone is the
size-limited Community Edition, which is not enough for most instances; a full license (free for
academics) is required. Any PuLP solver can be used instead (`--solver PULP_CBC_CMD`), but CBC is
much slower, so results within the time limits will differ.

Unless stated otherwise, the commands below run on CPU, from the root of the repository.

## Regenerating the figures from the provided data

```bash
cd memory-sweep
for p in 4 8 16; do python plot.py results/fig3/ngpus$p --throughput --nmb $((2*p)) --separate-legend --output fig3-${p}p.pdf; done
python plot_uniform.py results/fig6/per-microbatch/ngpus16 results/fig6/uniform/ngpus16 \
    --no-overhead --labels "Per-Mb" "Uniform" --throughput --nmb 32 --output fig6.pdf
cd ..

cd throughput
python plot.py 'results/ppcm21-*-16h100.with-adapipe.json' --metric throughput --sort-by config --err --output fig4.pdf
python plot_ablation.py 'results/ppcm21-*-16h100.with-adapipe.json' --metric throughput --sort-by config --err --output fig7.pdf
for m in 9-6 21 38-5; do for p in 4 8 16; do      # Figures 8 (9-6), 9 (21), 10 (38-5)
    python plot.py "results/ppcm$m-*-${p}h100.with-adapipe.json" --metric throughput --sort-by config --err --output fig-$m-${p}p.pdf
done; done
python bwd_remat_stats.py 'results/*.json'        # percentages quoted in Section 4.4
cd ..

cd solving-time
python plot.py results/fig5.json --output fig5.pdf
python plot.py results/fig11.json --unit mbs --output fig11.pdf
cd ..
```

## Re-running the experiments

### Figures 3 and 6 — ILP objective vs memory budget

The ILP is solved for decreasing memory budgets (128 s time limit, 1% gap) on profiling data of the
21B model with sequence length 4096. Figure 3 uses profilings with a single forward rematerialization
option per stage, Figure 6 uses profilings with AdaPipe's intermediate options (individual
attention/FFN blocks).

```bash
cd memory-sweep
# Figure 3
python simulate.py --profiled-dir inputs/single-option --output-dir out/fig3
# Figure 6
python simulate.py --profiled-dir inputs/intermediate-options --output-dir out/fig6/per-microbatch \
    --ngpus 16 --methods F-StageRemat-1f1b-1W FB-StageRemat-zbv-2W
python simulate.py --profiled-dir inputs/intermediate-options --output-dir out/fig6/uniform \
    --ngpus 16 --methods F-StageRemat-1f1b-1W FB-StageRemat-zbv-2W --uniform
```

A full sweep takes several hours (up to 128 s per point). To check a few points only, pass the memory
budgets (MB) explicitly, e.g. `--ngpus 4 --mems 29440 46240 63040`, and compare with `results/`.

The 1F1B-2W schedule (`megatron`) of the current ELF version interleaves the two stages of each
device differently than the version used for the paper, so the 1F1B-2W curve of Figure 3 is not
reproduced exactly (the estimated iteration time is a few percent lower). All other schedules are
identical to the ones used for the paper.

### Figures 5 and 11 — ILP solving time

Model 9.6B, sequence length 16384, 1 hour time limit, single thread.

```bash
cd solving-time
# Figure 5: increasing number of ranks (ZB-H1)
for p in 4 8 16 32; do
    python ../pipeline-ilps/exp_solving_time.py -c inputs/configs/ppcm9-6-s16384-${p}h100.json \
        -p inputs/profiled/ppcm9-6-s16384-h100-${p}stages.json -s zbh1 -t 3600 -n $p -S CPLEX_PY --output fig5.json
done
# Figure 11: 16 ranks, increasing number of micro-batches (ZB-V)
for m in 16 32 64 128 256; do
    python ../pipeline-ilps/exp_solving_time.py -c inputs/configs/ppcm9-6-s16384-16h100-${m}m.json \
        -p inputs/profiled/ppcm9-6-s16384-h100-32stages.json -s zbv -t 3600 -n $m -M 72000 -S CPLEX_PY --output fig11.json
done
python plot.py fig5.json --output fig5.pdf
python plot.py fig11.json --unit mbs --output fig11.pdf
```

The solver log is parsed with `orloge`; it is written to `/tmp/FB-StageRemat.log`, so do not run
several instances concurrently on the same machine.

### Figures 4, 7, 8, 9, 10 — measured throughput (GPU cluster)

The Snakemake workflow in `workflow/` runs, for each runfile in `workflow/runs/` (model, sequence
length, number of GPUs, number of micro-batches, memory budget and methods to evaluate):

1. profiling of the model split into as many stages as needed by each schedule, with
   `elf.profiling.profile_stage`, on 2 GPUs (`benchmarks/profile_stages.py`), and of its units for
   AdaPipe on 1 GPU (`benchmarks/profile_units.py`);
2. solving of the ILP and of the baselines for each schedule (`pipeline-ilps/runall.py`, CPU, 10 min limit);
3. execution of every solution on the target number of GPUs, 10 trials of 16 iterations
   (`benchmarks/bench_remat.py`, `benchmarks/bench_adapipe.py`). The rematerialization decisions are
   applied by `elf.scheduling.ilp.RematScheduler`.

Edit the `SLURM` section of `workflow/smk-config-h100.yaml` (account, venv path, GPU flags) for your
cluster. GPU jobs are submitted with `sbatch --wait` from the node running Snakemake.

```bash
cd workflow
snakemake -j 6 -k ../results/ilps/benchmarks/ppcm21-s{2048,4096,6144,8192}-{4,8,16}h100-mem72000.with-adapipe.json
snakemake -j 6 -k ../results/ilps/benchmarks/ppcm9-6-s{4096,8192,12288,16384}-{4,8,16}h100-mem72000.with-adapipe.json
snakemake -j 6 -k ../results/ilps/benchmarks/ppcm38-5-s{1024,2048,3072,4096}-{4,8,16}h100-mem72000.with-adapipe.json
```

`mem72000` is the memory budget given to the ILP (90% of the 80 GB H100 memory). The resulting
`results/ilps/benchmarks/*.with-adapipe.json` files have the same format as `throughput/results/` and
are plotted with the same commands. Failed configurations (infeasible solution, out-of-memory) are
dropped from the merged file and appear as missing bars.

The data in `throughput/results/` was measured with the version of ELF used for the paper. ELF has
evolved since then (rematerialization engine, memory model of the ILP, causal attention in the
benchmark Transformer, 1F1B-2W schedule), so new measurements will not match these values exactly.

## Method names

Result keys follow `<method>-<schedule>-<waves>W`:

| key | paper name |
|---|---|
| `F-StageRemat`, `FB-StageRemat` | ILP (ours), with forward-only (F) or forward + backward (F + B) rematerialization |
| `F-Greedy`, `FB-Greedy` | Greedy |
| `Uni-F-Remat`, `Uni-FB-Remat` | FullRemat |
| `NoRemat` | NoRemat |
| `AdaPipe` | AdaPipe |
| `1f1b-1W`, `megatron-2W`, `zbh2-1W`, `zbv-2W` | 1F1B, 1F1B-2W, ZB-H2, ZB-V |
