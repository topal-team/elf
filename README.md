# ELF – Efficient Deep Learning Framework

ELF is a lightweight research-oriented framework built on top of **PyTorch** that makes training very large neural networks on multi-GPU setups effortless. It automates everything that usually hurts when scaling a model beyond the memory of a single GPU: graph extraction, partitioning, device placement, scheduling, communication, and memory optimisation – all exposed through a single high-level API.

## Highlights

• **One-line pipeline parallelism** – wrap any `torch.nn.Module` inside `elf.Pipeline` and train it across any number of GPUs.
• **Automatic model partitioning** – integrates different model splitting algorithms, and respects manual splits when you prefer full control.
• **Static schedule zoo** – GPipe, 1F1B, Hanayo, Zero-Bubble family, full-remat and inference-only variants.
• **Data + pipeline parallelism** – mix pipeline stages (`pp`) with data-parallel replicas (`dp`) in the same job.
• **Fine-grained rematerialisation control** – pick a built-in schedule or inject your own policy to trade memory for extra compute.
• **Plugin registries** – add new schedulers, partitioners or tracers without touching the core code.

## Quick start

```python
import torch
from elf import Pipeline            # main entry point

torch.distributed.init_process_group("nccl")

model   = MyBigModel()
sample  = torch.randn(input_shape, device='cuda')   # only needed for profiling
inputs  = ...
targets = ...

pipe = Pipeline(model, sample)               # pass placement / partitioner / scheduler as needed

loss_fn = torch.nn.CrossEntropyLoss()
y, loss = pipe(inputs, targets, loss_fn)   # forward + backward (+ DP gradient sync)

# usual optimizer step
optimizer.step()
pipe.zero_grad()
```

Call `pipe.clear()` once you are done to gracefully destroy the underlying process-groups.\
Some examples can be found under `examples/` for more details and use cases.

## The Pipeline API

`Pipeline` is a thin wrapper around your `nn.Module`.  The most useful kwargs are:

• **placement** – list of CUDA ranks (or `"auto"`) describing where each stage runs.
• **partitioner** – registry key or callable used to cut the graph (set to `False` if you already partitioned the model yourself).
• **scheduler** – registry key or callable that returns a static list of operations for every micro-batch.
• **dp** – integer giving the data-parallel replication factor.

The full argument list is defined in [the documentation](#docs).


## Registries: plug & play algorithms

ELF exposes three global registries in `elf.registry`:

```python
from elf.registry import SCHEDULERS, PARTITIONERS, TRACERS
```

Register a new component by key:

```python
def my_partitioner(graph, times, memories, n_parts):
    ...

PARTITIONERS.register("my_algo", my_partitioner, description="Algo from paper ...")
```

Then simply reference it when building a pipeline: `Pipeline(..., partitioner="my_algo")`.

The signature of functions expected in the registry are detailed in `elf/registry.py`

## Process topologies

`Placement.default(scheduler, pp)` gives a good default mapping, but you can pass any explicit list, enabling exotic layouts such as:

```python
placement = [0,1,2,3, 3,2,1,0]   # bidirectional pipeline for Hanayo / ZBV
```

## Environment variables

- ``ELF_TIMINGS``: Accurate time measurements in ``detailed_stats`` field of the ``Pipeline`` object after an iteration. May affect performance.
- ``ELF_MEMORY``:  Accurate kept and peak memory measurements in ``detailed_stats`` field of the ``Pipeline`` object after an iteration. May affect performance.
- ``ELF_TIMEOUT``: Number of seconds to wait for before shutting down process groups. (passed to NCCL watchdog)

## Docs

The full documentation can be generated with Sphinx. Go to `docs/` and run `make html`.

## Testing & benchmarks

• Run all tests: `./tests/test.sh` (a few require ≥2 GPUs).
• Performance scripts live in `benchmarks/`.
• ILP-based experiments are located in `ilps`.

## Environment setup on clusters

CUDA/NCCL versions and SLURM setups for some HPC clusters are documented in the `docs/` folder:

• [Jean-Zay](docs/jean-zay.md)
• [Helios](docs/helios.md)