## How to run the ILPs benchmarks

All scripts expect a model json configuration file, containing a "model" entry with "hidden_dim", "seq_len", "n_heads" and "dropout" values. Examples can be found in ``configs/``. The metadata entry is optional.\
Since the ILP formulations are in another repo (``pipeline-ilps``) you should create a link to the ``runall.py` file that is used to solve the ILPs.
```bash
ln -s ~/pipeline-ilps/runall.py ilps/runall.py
```


### Experiment 1: Iteration time vs Model size

The main script is ``run_ilps_benchmark.sh``, it should be run directly on a gpu node.\
Usage:
```bash
./ilps/run_ilps_benchmark.sh CONFIG_FILE [ngpus] [min_blocks] [max_blocks] [step]
```

If ngpus is higher than the number of GPUs available on the nodes, the script will not run the final benchmark, but instead give you the command to execute it.\
Otherwise the result will be written to ``results/bench-ilps-{config-file-name}.json``.

### Experiment 2: Iteration time vs Sequence length

The ``run_ilps_seqlen_benchmark.sh`` script is a bit different: you can run it directly on the front node and it will submit a bunch of jobs in parallel.\
Usage:
```bash
./ilps/run_ilps_seqlen_benchmark.sh CONFIG_FILE [ngpus] [min_seqlen] [max_seqlen] [step]
```

The results will be located in ``results/seqlen_benchmarks``. You can run the command printed by the file to merge the results together once all jobs are finished. The final result file will then be ``results/seqlen_benchmarks/summary.json``.
