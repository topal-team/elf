#!/bin/bash

#SBATCH --constraint='sirocco'
#SBATCH --job-name=multinode
#SBATCH --exclude=sirocco[01-06,10-15,17,24-25]

if [[ $SLURM_NTASKS -gt 1 ]] ; then
    nodes=($( scontrol show hostnames $SLURM_JOB_NODELIST ))
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    echo Head node ip : $head_node_ip $head_node
    options="--rdzv-id $RANDOM --rdzv-backend c10d --rdzv-endpoint $head_node_ip:26500"
else
    options=""
fi
   
srun singularity exec --nv /beegfs/aaguilam/images/nanotron.sif \
torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc-per-node $SLURM_NTASKS_PER_NODE \
${options} $1
