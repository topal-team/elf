#!/bin/bash

## JOB INFO
#SBATCH --job-name=compare_attention
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

## NODE CONFIGURATION
#SBATCH --constraint=h100
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread

## JOB ACCOUNTABILITY
#SBATCH --account=gdh@h100
#SBATCH --time=00:20:00


# set -x
## ENV ACTIVATION
module purge
module load arch/h100
module load nvidia-nsight-systems/2024.7.1.84

source $HOME/idrenv-gdh.sh 

source $WORK/venvs/elf/bin/activate

CODEDIR=$HOME/Workspace/elf-dev/benchmarks

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# torchun --nnodes=1 --nproc_per_node=4 "${CODEDIR}/train_diffusion_elf.py"
# srun torchrun --nnodes=2 --nproc_per_node=4 --rdzv-id 55 --rdzv-backend c10d --rdzv-endpoint "${nodes[0]}"  "${CODEDIR}/compare_attentions.py" 

# nsys profile --trace cuda,cudnn,cublas,nvtx --output profstats --stats=true --cudabacktrace=all --capture-range=cudaProfilerApi --capture-range-end=stop torchrun --nproc-per-node 4  "${CODEDIR}/compare_attentions.py"
# nsys profile --trace cuda,cudnn,cublas,nvtx --output profstats --stats=true --cudabacktrace=all --capture-range=cudaProfilerApi --capture-range-end=stop torchrun --nproc-per-node 4  "${CODEDIR}/compare_attentions.py"

srun nsys profile --trace cuda,cudnn,cublas,nvtx --output profstats%p --stats=true --cudabacktrace=all --capture-range=cudaProfilerApi --capture-range-end=stop torchrun --nnodes=2 --nproc_per_node=4 --rdzv-id 55 --rdzv-backend c10d --rdzv-endpoint "${nodes[0]}"  "${CODEDIR}/compare_attentions.py" 
