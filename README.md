## Script usage for PlaFRIM

Use script ``start.sh`` to start multi-nodes tasks.\
Usage:
```bash
sbatch --nodes=? ./start.sh {your_script}.py
```
By default, the script will use every GPU on every node, and start one process per gpu.

## For Jean-Zay

I recommend allocating a node in interactive mode !
Then, you can use :

```bash
singularity exec --nv --bind $(pwd):$(pwd) $SINGULARITY_ALLOWED_DIR/pipe.sif torchrun --nnodes 1 --nproc-per-node 4 --standalone -- {your_script}.py
```

## How it works

### Create the pipeline

The object ``Pipeline`` from ``pipeline`` provides a simple API to automatically take care of everything.
```py
from pipeline import Pipeline
sample = torch.randn(..., device = 'cuda')
pipe = Pipeline(model, sample)
y, loss = pipe(inputs, targets, loss_fn)
```

Note that ``sample`` is only necessary if you use automatic partitioning, as it is used for profiling.\
There are several arguments to modify its behaviour :
- ``placement`` specifies the rank of each model block.
- ``partition`` can be set to ``None`` to disable automatic partition. This is useful in case you already partitioned your model yourself. Each part should be placed on the right device.
- ``schedule`` modifies the schedule algorithm to use. Currently, AFAB from Gpipe, 1F1B and Hanayo are supported.

### Write your own schedule

You can define your own schedule if you want to perform tests or use an unimplemented one. In order to do that, you simply have to write a function that takes as argument a ``placement`` and a number of micro batches ``n_micro_batches``, and returns the right sequence of operations (see the ``Operation`` class in ``graph.py``). Then, register it in the ``Pipeline`` class in ``pipeline.py``.

### Change the pipeline behaviour

The options can be anything that modifies the behaviour of the operation, as long as the corresponding function in ``pipeline.py`` is modified to take it into account. See remat for an example. Currently supported :

- Rematerialization (``{"remat": True}``)
- Offloading (``{"offload": True}``)

### Model partitioning

Different partition scheme are available. All of them rely on ``torch.fx.symbolic_trace``, so if your model cannot be traced properly you will probably have to partition it yourself.
The different partition modes are:
- ``default``: Naive graph partition that tries to balance computation times for each part
- ``constrained``: Same as default, but with a hard constraint on each part to have exactly 1 input tensor and 1 output tensor
- ``metis``: Call [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) to optimize the partition. Needs ``gpmetis`` to be installed. Currently has some known issues of cycles.
- ``dagP``: Call [dagP](https://github.com/GT-TDAlab/dagP/) to partition. Needs ``rMLGP`` installed. 
