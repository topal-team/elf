## Start on Jean-Zay

Use the script ``multinode.sh``. You can provide the number of GPUs and nodes to use. If your job is long, don't forget to add the ``--time hh:mm:ss`` option.

Example:
```bash
sbatch --gpus 16 --nodes 4 --time 01:00:00 multinode.sh script.py
```

You can provide additional directory bindings in the script if needed (for instance a data folder).

## How it works

### Create the pipeline

The object ``Pipeline`` from ``pipeline`` provides a simple API to automatically take care of everything.
```py
from pipeline import Pipeline
sample = torch.randn(..., device = 'cuda')
pipe = Pipeline(model, sample)
y, loss = pipe(inputs, targets, loss_fn)
pipe.clear()
```

A full example can be found in ``example.py``.

Note that ``sample`` is only necessary if you use automatic partitioning, as it is used for profiling. Also, it is not used on processes other than the first one of the pipeline.\
There are several arguments to modify its behaviour :
- ``placement`` specifies the rank of each model block.
- ``partition`` can be set to ``None`` to disable automatic partition. This is useful in case you already partitioned your model yourself. Each part should be placed on the right device.
- ``schedule`` modifies the schedule algorithm to use. Currently, AFAB from Gpipe, 1F1B and Hanayo are supported.
- ``dp`` is the data parallelism degree.

### Write your own schedule

You can define your own schedule if you want to perform tests or use an unimplemented one. In order to do that, you simply have to write a function that takes as argument a ``placement`` and a number of micro batches ``n_micro_batches``, and returns the right sequence of operations (see the ``Operation`` class in ``graph.py``). Then, register it in the ``Pipeline`` class in ``pipeline.py`` (``_get_scheduler``).

### Change the pipeline behaviour

The options can be anything that modifies the behaviour of the operation, as long as the corresponding function in ``pipeline.py`` is modified to take it into account. See remat for an example. Currently supported :

- Rematerialization (``{"remat": True}``)
<!-- - Offloading (``{"offload": True}``) -->

### Model partitioning

Different partition scheme are available. All of them rely on ``torch.fx.symbolic_trace`` or ``torch.export`` (whichever works, default is fx). If your model cannot be traced properly you will have to partition it yourself.
The different partition modes are:
- ``naive``: Naive graph partition that tries to balance computation times for each part
- ``constrained``: Same as default, but with a hard constraint on each part to have exactly 1 input tensor and 1 output tensor
- ``metis``: Call [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) to optimize the partition. Needs ``gpmetis`` to be installed. Currently has some known issues of cycles.
- ``dagP``: Call [dagP](https://github.com/GT-TDAlab/dagP/) to partition. Needs ``rMLGP`` installed. 

### Data parallelism

You can run multiple pipelines at once with different data by specifying the argument ``dp`` of the ``Pipeline`` object. Each pipeline will be replicated ``dp`` times. Note that the user needs to handle the data loading and distribute it among the pipelines, for instance with ``torch.utils.data.distributed.DistributedSampler``.
