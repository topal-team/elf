## How to run the ILPs benchmarks

All scripts expect a model json configuration file, containing a "model" entry with "hidden_dim", "seq_len", "n_heads" and "dropout" values. Examples can be found in ``configs/``. The metadata entry is optional.\
The scripts assume the file structure to be:
```
./
    ilps/
        configs/
            - config1.json
            - config2.json
    results/
        profiling/
        regression/
        ilps-solutions/
    elf/
```

All paths should be created automatically if needed by the scripts.

### Experiment 1: Iteration time vs Model size

The main script is ``run_ilps_benchmark.sh``, it should be run directly on a gpu node.\
Usage:
```bash
./ilps/run_ilps_benchmark.sh --config CONFIG_FILE [--ngpus N] [--min-blocks N] [--max-blocks N] [--step N] [--scheduler NAME] [--memgpu N] [--account NAME] [--constraint NAME] [--regression-file FILE] [--sdp-backend BACKEND]
```

The script will create a new sbatch script that you can run to start all benchmark jobs. The path is ``results/run_benchmarks_<CONFIG>.sh``.\
After the jobs are finished, the results will be stored in ``results/bench-ilps-<CONFIG>.json``.\
*\<CONFIG> is the name of the provided config file without extension.*


### Experiment 2: Iteration time vs Sequence length

The ``run_ilps_seqlen_benchmark.sh`` script is a bit different, since it needs to run multiple profilings; starting the script will run some jobs on GPU nodes. Each one of these jobs will create a new config file under ``ilps/configs/seqlen_configs`` and generate the ILP solutions + baselines. Then, the same sbatch script as before will be generated under the path ``results/seqlen_benchmarks/run_benchmarks_<CONFIG>.sh``. The results however will have to be merged into one file once all jobs are finished, with the command  ``results/seqlen_benchmarks/summary.json``.\
Usage:
```bash
./ilps/run_ilps_seqlen_benchmark.sh --config CONFIG_FILE [--ngpus N] [--min-seqlen N] [--max-seqlen N] [--step N] [--nblocks N] [--scheduler NAME] [--memgpu N] [--account NAME] [--constraint NAME] [--sdp-backend BACKEND] [--precision PRECISION] 
```