#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --time=00:20:00
#SBATCH --output=out_files/out.out
#SBATCH --error=out_files/err.err
#SBATCH --account=tutorial

# run as `sbatch --nodes 2 --gres=gpu:4 example.sh` 
# IMPORTANT: load the modules for machine learning tasks and libraries
module load ML-bundle/24.06a
module load NVHPC/24.5-CUDA-12.4.0

VENV_PATH=${SCRATCH}/venvs/venv
CODE_PATH=$HOME/topal-internship/
PYSCRIPT_PATH=$HOME/topal-internship/example.py

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

cd $CODE_PATH/helios

if [ "$SLURM_NNODES" -gt 1 ]; then
    echo "This job is running on more than 1 node."

    # Set environment variables for PyTorch distributed
    # Use the first node as the master
    # Generate a random port in the range [10000, 65535]
    export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)  
    export MASTER_PORT=$(( ( RANDOM % 55536 ) + 10000 ))                      

    # Print selected MASTER_ADDR and MASTER_PORT for debugging
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"

    srun torchrun --nnodes $SLURM_NNODES \
            --nproc_per_node $SLURM_GPUS_ON_NODE\
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT ${PYSCRIPT_PATH}
else
    echo "This job is running on a single node."
    echo ${SLURM_NNODES} ${SLURM_GPUS_ON_NODE}
    torchrun --nnodes 1 --nproc_per_node  ${SLURM_GPUS_ON_NODE} ${PYSCRIPT_PATH} 
fi

# torchrun --nnodes=2 --nproc_per_node=4 $HOME/topal-internship/example.py

# Run torchrun with the appropriate arguments
# srun torchrun --nnodes=2 \
#          --nproc_per_node=4\
#          --rdzv_backend=c10d \
#          --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#          $HOME/topal-internship/example.py
