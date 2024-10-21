#!/bin/bash -l

#SBATCH -A tutorial
#SBATCH -t 01:00:00
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --output out.slurm

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo "Master node : " ${nodes[0]}

OPTIONS="--nnodes $SLURM_NNODES "
OPTIONS+="--nproc-per-node $SLURM_GPUS_ON_NODE "

if [[ $SLURM_NNODES -gt 1 ]]; then
    OPTIONS+="--rdzv-id $SLURM_JOBID "
    OPTIONS+="--rdzv-backend c10d "
    OPTIONS+="--rdzv-endpoint ${nodes[0]} "
else
    OPTIONS+="--standalone "
fi

module load ML-bundle/24.06a GCCcore/13.2.0 Python/3.11.5 CUDA/12.4.0 > /dev/null 2>&1

source $GROUPS_STORAGE/elf/venv/bin/activate

srun torchrun $OPTIONS -- $*

