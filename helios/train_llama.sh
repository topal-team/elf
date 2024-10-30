#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --nodes=2
##SBATCH --ntasks-per-node=1
#SBATCH --time=00:60:00
#SBATCH --output=out_files/out.out
#SBATCH --error=out_files/err.err
#SBATCH --gres=gpu:4
#SBATCH --account=tutorial

# IMPORTANT: load the modules for machine learning tasks and libraries
module load ML-bundle/24.06a
module load NVHPC/24.5-CUDA-12.4.0

VENV_PATH=${SCRATCH}/venvs/venv
CODE_PATH=$HOME/topal-internship/
PYSCRIPT_PATH=$HOME/topal-internship/scripts/train_llama.py
DATA_PATH=${GROUPS_STORAGE}/elf/data/c4-realnewslike
TOKENIZER_PATH=${GROUPS_STORAGE}/elf/tokenizer/

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

# Set environment variables for PyTorch distributed
# export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)  # Use the first node as the master
# export MASTER_PORT=$(( ( RANDOM % 55536 ) + 10000 ))                      # Generate a random port in the range [10000, 65535]

# Print selected MASTER_ADDR and MASTER_PORT for debugging
# echo "MASTER_ADDR: $MASTER_ADDR"
# echo "MASTER_PORT: $MASTER_PORT"

cd  ${CODE_PATH}

if [ "$SLURM_NNODES" -gt 1 ]; then
    echo "This job is running on more than 1 node."
    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    srun torchrun --nnodes $SLURM_NNODES \
            --nproc_per_node $SLURM_GPUS_ON_NODE\
            --rdzv_backend=c10d \
            --rdzv_endpoint ${nodes[0]} \
            ${PYSCRIPT_PATH} --dataset_path ${DATASET_PAT} --tokenizer_path ${TOKENIZER_PATH} -dp 2 -pp 4
else
    echo "This job is running on a single node."
    torchrun --nnodes 1 --nproc_per_node  $SLURM_GPUS_ON_NODE ${PYSCRIPT_PATH} --dataset_path ${DATA_PATH} --tokenizer_path $GROUPS_STORAGE/elf/tokenizer/ -dp 2 -pp 4
fi
