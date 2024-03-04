## Script usage for PlaFRIM

Use script ``start.sh`` to start multi-nodes tasks.\
Usage:
```
sbatch --nodes=? ./start.sh your_script.py
```
By default, the script will use every GPU on every node, and start one process per gpu.

