## How it works

### Create the pipeline

The object ``Pipeline`` from ``pipeline`` provides a simple API to automatically take care of everything.
```py
from elf import Pipeline
sample = torch.randn(..., device = 'cuda')
pipe = Pipeline(model, sample)
y, loss = pipe(inputs, targets, loss_fn)
pipe.clear()
```

A full example can be found in ``examples/basic.py``.

Note that ``sample`` is only necessary if you use automatic partitioning, as it is used for profiling. Also, it is not used on processes other than the first one of the pipeline.\
There are several arguments to modify its behaviour :
- ``placement`` specifies the rank of each pipeline stage.
- ``partitioner`` can be set to ``False`` to disable automatic partition. This is useful in case you already partitioned your model yourself. Each part should be placed on the right device. Otherwise, you can choose between partitioners described below.
- ``schedule`` modifies the schedule algorithm to use. Currently, Gpipe (`"afab"`), 1F1B (`"1f1b"`) and Hanayo (`"hanayo"`) are supported.
- ``dp`` is the data parallelism degree. One pipeline will be replicated ``dp`` times.

### Write your own schedule

You can define your own schedule if you want to perform tests or use an unimplemented one. In order to do that, you simply have to write a function that takes as argument a ``placement``, a number of micro batches ``n_micro_batches``, and the signature of each block, to return the right sequence of operations (see the ``Operation`` class in ``scheduling.py``). Then, register it in the ``Pipeline`` class in ``pipeline.py`` (``_get_scheduler``).

### Change the pipeline behaviour

The options can be anything that modifies the behaviour of an operation (forward, send, recv, ..). The corresponding function in ``block.py`` needs to be modified to take it into account. See remat for an example.

Each supported option is in the enumeration ``OpOptions`` in ``scheduling.py``.

### Model partitioning

Different partition scheme are available. All of them rely on ``torch.fx.symbolic_trace`` or ``torch.export`` (whichever works, default is fx). If your model cannot be traced properly you will have to partition it yourself.
The different partition modes are:
- ``naive``: Naive graph partition that tries to balance computation times for each part
- ``constrained``: Same as default, but with a hard constraint on each part to have exactly 1 input tensor and 1 output tensor
- ``metis``: Call [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) to optimize the partition. Needs ``gpmetis`` to be installed. Currently has some known issues of cycles.
- ``dagP``: Call [dagP](https://github.com/GT-TDAlab/dagP/) to partition. Needs ``rMLGP`` installed. 

### Data parallelism

You can run multiple pipelines at once with different data by specifying the argument ``dp`` of the ``Pipeline`` object. Each pipeline will be replicated ``dp`` times. Note that the user needs to handle the data loading and distribute it among the pipelines, for instance with ``torch.utils.data.distributed.DistributedSampler``.

## Running tests and benchmarks

Environment setup on some clusters is described in `docs`.
- [Helios](docs/helios.md)

### Tests

Run all tests:
```bash
./tests/test.sh
```
Some of them require at least 2 GPUs.

### Benchmarks

Some benchmarks are available in ``benchmarks/``. They are used to measure and compare the performance of the pipeline.
