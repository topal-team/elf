#!/bin/bash

#SBATCH --constraint='sirocco'
#SBATCH --job-name=multinode
#SBATCH --exclusive
#SBATCH --exclude=sirocco[01-07,10-15,17,24-25]

nodes=$( scontrol show hostnames $SLURM_JOB_NODELIST )
if [[ $SLURM_JOB_NUM_NODES -gt 1 ]] ; then
    nodes_array=($nodes)

    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    echo Head node ip : $head_node_ip $head_node
    options="--rdzv-id $RANDOM --rdzv-backend c10d --rdzv-endpoint $head_node:26501"
else
    options="--standalone"
fi

echo Options : ${options}
echo Using ${SLURM_JOB_NUM_NODES} nodes \($nodes\) and every GPU on every node.

srun singularity exec --nv /beegfs/aaguilam/images/nanotron.sif \
torchrun --nnodes ${SLURM_JOB_NUM_NODES} --nproc-per-node=gpu \
${options} $1
