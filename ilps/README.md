## Profiling format

The ILPs need information about the time and memory statistics of your model. This information must respect a specific format:
```json
[
    {
        "T": [Compute time F, Compute time B, Compute time W],
        "M": [Memory kept F, Memory kept B, Memory kept W],
        "Tcomm": Time taken to send the output,
        "Mparams": Amount of memory to store parameters,
        "forward_remat_options": [
            {
                "name": Option name,
                "overhead": Additional time needed during B,
                "memory freed": Amount of memory deleted after F
            },
            {
                ...
            }
        ],
        "backward_remat_options": [
            {
                "name": Option name,
                "overhead": Additional time needed during W,
                "memory freed": Amount of memory deleted after B
            },
            {
                ...
            }
        ]
    },

    {
        "T": ...,
        "M": ...,
        ....
    }
]
```

Let's break it down:
the file should contain a list of subconfigs. Each subconfig corresponds to one stage of the pipeline (one part of your model). It has a profiled execution time and an amount of memory kept for all 3 operations (F, B, W). Tcomm is the profiled communication size for the output of this stage (not the input!). Mparams is the amount of memory taken by the parameters.

Then, it can have rematerialization options. There are two kinds of options, forward and backward. They have exactly the same syntax. A rematerialization option has a name (not very useful, just use different names for all), a time overhead induced by the recomputation, and an amount of memory deleted compared to storing everything.\
Note that the options "recompute everything" and "recompute nothing" are added automatically, no need to include them.\
The units for time and memory do not matter, as long as they're the same for every value.

The file ``params.py`` provides a utility function, ``get_params_from_config``, to parse this file into a python object that uses a clearer structure. For instance, rematerialization options use a unified ``RematOption`` representation. Some values are also added:
- $p$ the number of gpus
- $m$ the number of micro-batches
- $Mgpu$ the memory budget on one GPU
- $sched$ the order of operations, as a list of lists of characters in $\{f, b, w\}$

You are free to change these values after parsing the parameters. Just remember to re-generate the schedule if you change the number of processors or micro-batches.

For ILPs that optimize load balancing (Balance and BlockRemat), an additional parameters $n$ can be passed to that utility to specify the number of blocks. In that case, your file should provide profiling values for one block (but still have as many subconfigs as the number of stages). These values will then be scaled according to the load balancing. For instance, if you have 4 GPUs and an homoegenous Transformer of 32 blocks for which you want to use BlockRemat, your file should contain 4 subconfigs with the statistics of 1 block each. Then you can give $n=32$ to the ``get_params_from_config`` function.\
Warning: do not use load balancing ILPs and/or the $n$ parameter with heterogeneous models.

For an example, please look at the output file after the regression step of the ILPs benchmark in ``ilps/run_ilps_benchmark.sh``.

## Running the ILPs

Once your profiling file is ready, you can solve the ILPs. They all use the same interface, defined in ``base.py:LpProblem``. The basic usage is:
```py
params = get_params_from_config(myfile, nprocs)
problem = RematProblem("StageRemat", params)
status = problem.solve()
solution = problem.get_solution()
```
It is then up to you to use this solution as an execution plan for the pipeline. An example of that is in ``benchmarks/ilp_schedulers.py``. You can also refer to the documentation of ``elf/scheduling.py:OpOptions`` and the computation functions in ``elf/block.py`` to understand how to define the rematerialization options.

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