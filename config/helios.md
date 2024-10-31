## Setup on helios

### General environment


Allocate a node in interactive mode:

```bash
salloc --time 01:00:00 --nodes=1 --gpus-per-node 4 --exclusive -A tutorial -p gpu srun --interactive --preserve-env --pty /bin/bash -l
```

Load the required modules:
```bash
module load ML-bundle/24.06a
module load GCCcore/13.2.0 Python/3.11.5 CUDA/12.4.0
```

Use either a python environment, or a container:

#### Python venv
Create it in scratch:
```bash
python -m venv $SCRATCH/venv
source $SCRATCH/venv/bin/activate
pip install --no-cache torch==2.5.0.cyf2-cu124.post2 torchvision==0.19.0+cu124
cd ~/topal-internship/
pip install --no-cache -r requirements.txt
```

Then activate it with:
```bash
source $SCRATCH/venv/bin/activate
```

#### Container

The image is available at `$GROUPS_STORAGE/elf/images/pipe.sif`. Either copy it to your scratch space, or use it directly.

```bash
apptainer exec --nv path/to/image/pipe.sif {your_command}
```

Add METIS to your path to be able to use it:
```bash
export PATH=$PATH:$GROUPS_STORAGE/elf/packages/local/bin
```

#### Data management

Most script using data take the `--dataset_path` (or `-d`) argument. I recommend always specifying it to avoid getting data downloaded to your home directory. Some datasets are already available in `$GROUPS_STORAGE/elf/data`, including tokenized versions (tokenizing can be very long).

## Batch jobs

A script `helios.sh` is provided in root directory. It can be run like:
```bash
sbatch --gpus 8 helios.sh script.py
```

### Mandatory options to write your sbatch script

```bash
#!/bin/bash -l

#SBATCH -A tutorial
#SBATCH -t 01:00:00 # Default is 10 minutes
#SBATCH -p gpu
```
