#!/bin/bash

#SBATCH --account hyb@v100
#SBATCH --exclusive
#SBATCH --out out.slurm

# Get the number of nodes and GPUs per node from SLURM# Get the master node's hostname
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun singularity exec --nv --bind \
    $(pwd):$(pwd),$SCRATCH:/data,$DSDIR/HuggingFace_Models:/models \
    $SINGULARITY_ALLOWED_DIR/pipe.sif \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=$SLURM_GPUS_ON_NODE \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:29500 \
    -- scripts/train_llama.py $*