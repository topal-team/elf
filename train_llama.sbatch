#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --time=00:60:00
#SBATCH --output=/net/scratch/hscra/project/tutorial/tutorial050/slurm_logs/%J.out
#SBATCH --error=/net/scratch/hscra/project/tutorial/tutorial050/slurm_logs/%J.err
#SBATCH --account=tutorial

# IMPORTANT: load the modules for machine learning tasks and libraries
module load ML-bundle/24.06a
module load NVHPC/24.5-CUDA-12.4.0
HOME="/net/home/project/tutorial/tutorial050/"
SCRATCH="/net/scratch/hscra/project/tutorial/tutorial050/"
VENV_PATH=$SCRATCH/venvs/venv
CODE_PATH=$HOME/topal-internship/topal-internship/
PYSCRIPT_PATH=$HOME/topal-internship/train_llama.py

if [ -d "${VENV_PATH}" ]; then
    echo "The environment ${VENV_PATH} exists."
else
    echo "The environment ${VENV_PATH} does not exist."
    
    python3 -m venv ${VENV_PATH}
    echo "Created the environment ${VENV_PATH}."

    # install one of torch versions available at Helios wheel repo
    python3 -m pip install --no-cache-dir torch==2.4.0+cu124.post2 torchvision==0.19.0+cu124
    python3 -m pip install psutil pandas pyarrow
    python3 -m pip install datasets transformers
fi

# Activate the virtual environment
source ${VENV_PATH}/bin/activate
echo "Activated the environment ${VENV_PATH}."

echo "python3 --version"
python3 -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"

cd  ${CODE_PATH}
echo "HOHO"
if [ "$SLURM_NNODES" -gt 1 ]; then
    echo "This job is running on more than 1 node."
    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    srun torchrun --nnodes $SLURM_NNODES \
            --nproc_per_node $SLURM_GPUS_ON_NODE\
            --rdzv_backend=c10d \
            --rdzv_endpoint ${nodes[0]} \
            ${PYSCRIPT_PATH} ${PYARGS}
else
    echo "This job is running on a single node."
    torchrun --nnodes 1 --nproc_per_node  $SLURM_GPUS_ON_NODE ${PYSCRIPT_PATH} ${PYARGS}
fi
