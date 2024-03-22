## Script usage for PlaFRIM

Use script ``start.sh`` to start multi-nodes tasks.\
Usage:
```
sbatch --nodes=? ./start.sh your_script.py
```
By default, the script will use every GPU on every node, and start one process per gpu.

## Pipeline Engine

In the file ``engine.py`` you can find the ``StageScheduler``, an engine made to execute a custom pipeline schedule on an arbitrary list of blocks placed on different devices. It takes as inputs a list of blocks, which are assumed to be placed on the current cuda device for each process, and a schedule. The schedule is a list of tuples (block_id, operation) where operation is from the enum Operations. The available ones are:
- Receive forward (``RECV_FORWARD``)
- Compute forward (``FORWARD``)
- Send forward (``SEND_FORWARD``)
- Receive backward (``RECV_BACKWARD``)
- Compute backward (``BACKWARD``)
- Send backward (``SEND_BACKWARD``)