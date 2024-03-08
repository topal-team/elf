#!/bin/bash

#SBATCH --constraint='sirocco'
#SBATCH --job-name=multinode
#SBATCH --exclusive
#SBATCH --exclude=sirocco[01-05]

nodes=$( scontrol show hostnames $SLURM_JOB_NODELIST )
if [[ $SLURM_JOB_NUM_NODES -gt 1 ]] ; then
    nodes_array=($nodes)

    head_node=${nodes_array[0]}

    echo Head node : $head_node
    options="--rdzv-id $RANDOM --rdzv-backend c10d --rdzv-endpoint $head_node:26501"
fi

echo Options : ${options}
echo Using ${SLURM_JOB_NUM_NODES} nodes \($nodes\) and every GPU on every node.

srun singularity exec --nv /beegfs/aaguilam/images/nanotron.sif \
torchrun --nnodes ${SLURM_JOB_NUM_NODES} --nproc-per-node=2 ${options} $1
