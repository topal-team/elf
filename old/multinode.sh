#!/bin/bash

#SBATCH --account hyb@v100
#SBATCH --exclusive
#SBATCH --output out.slurm

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo "Master node : " ${nodes[0]}

bindings="$(pwd):$(pwd)"

srun singularity exec --nv --bind $bindings $SINGULARITY_ALLOWED_DIR/pipe.sif \
     torchrun --nnodes $SLURM_NNODES --nproc-per-node $SLURM_GPUS_ON_NODE --rdzv-id $SLURM_JOBID --rdzv-backend c10d --rdzv-endpoint ${nodes[0]} -- \
     $*
