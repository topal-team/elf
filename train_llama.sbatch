#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --time=00:60:00
#SBATCH --output=/net/scratch/hscra/project/tutorial/tutorial050/slurm_logs/%J.out
#SBATCH --error=/net/scratch/hscra/project/tutorial/tutorial050/slurm_logs/%J.err
#SBATCH --account=tutorial

export NCCL_SOCKET_IFNAME=hsn
export CXI_FORK_SAFE="1"
export FI_CXI_DISABLE_CQ_HUGETLB="1"
export FI_MR_CACHE_MONITOR="userfaultfd"
export CXI_FORK_SAFE_HP="1"
export NCCL_CROSS_NIC="1"
export FI_CXI_DISABLE_HOST_REGISTER="1"
export FI_HMEM_CUDA_USE_GDRCOPY=1
export FI_HMEM_CUDA_ENABLE_XFER=1

export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL="PHY"
export NCCL_CUMEM_ENABLE=0

# module load ML-bundle/24.06a
# module load NVHPC/24.5-CUDA-12.4.0

module load ML-bundle/24.06a GCCcore/13.2.0 Python/3.11.5 CUDA/12.4.0

# VENV_PATH="$SCRATCH/venvs/venv"
VENV_PATH="$GROUPS_STORAGE/elf/adrien-venv"
PYARGS=("$@")
PYARGS+=" --slurm_jobid $SLURM_JOBID"

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
echo "VENV_PATH ${VENV_PATH}"
source ${VENV_PATH}/bin/activate
echo "Activated the environment ${VENV_PATH}"
which torchrun

python3 -m pip install configmypy 

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
srun hostname
srun echo $PYARGS
if [ "$SLURM_NNODES" -gt 1 ]; then
    ranks=($(scontrol show hostnames ${SLURM_JOB_NODELIST}))
    echo "This job is running on more than 1 node."
    srun echo  ${ranks[0]}
    srun ${VENV_PATH}/bin/torchrun --nnodes $SLURM_NNODES \
        --nproc-per-node $SLURM_GPUS_ON_NODE \
        --rdzv-id $SLURM_JOBID \
        --rdzv-backend c10d \
         --rdzv-endpoint ${ranks[0]} \
         ${PYARGS[*]}
else
    echo "This job is running on a single node."
    torchrun --nnodes 1 --nproc-per-node  ${SLURM_GPUS_ON_NODE} ${PYSCRIPT_PATH} ${PYARGS}
fi
