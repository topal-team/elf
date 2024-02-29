#!/bin/bash

#SBATCH --constraint='sirocco'
#SBATCH --job-name=multinode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --exclude=sirocco[01-06,10-15,17,24-25]

nodes=($( scontrol show hostnames $SLURM_JOB_NODELIST ))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Head node ip : $head_node_ip $head_node

srun singularity exec --nv /beegfs/aaguilam/images/nanotron.sif \
torchrun --nnodes $SLURM_NNODES --nproc-per-node $SLURM_NTASKS_PER_NODE --rdzv-id $RANDOM --rdzv-backend c10d \
--rdzv-endpoint $head_node_ip:26500 $1
