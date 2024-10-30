#!/bin/bash -l

#SBATCH -A tutorial
#SBATCH -t 01:00:00
#SBATCH -p gpu
#SBATCH --exclusive
#SBATCH --output out.slurm

args=("$@")
remove_arg() {
    local arg_to_remove="$1"
    for i in "${!args[@]}"; do
        if [[ "${args[i]}" == "$arg_to_remove" ]]; then
            unset 'args[i]'
        fi
    done
}

CMD=""
if [[ "$@" == *"--container"* ]]; then
    export TMPDIR=$MEMFS/tmp
    CMD+="apptainer exec --nv $SCRATCH/apptainer/images/pipe.sif "
    remove_arg "--container"
else
    module load ML-bundle/24.06a GCCcore/13.2.0 Python/3.11.5 CUDA/12.4.0 > /dev/null 2>&1
    source $SCRATCH/venv/bin/activate
    echo "Activated virtual environment."
    python -c "import torch; print('Torch version:', torch.__version__)"
fi

if [[ "$@" == *"--no-multi"* ]]; then
    CMD+="python "
    remove_arg "--no-multi"
elif [[ "$@" == *"--no-python"* ]]; then
    remove_arg "--no-python"
else
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

    CMD+="torchrun $OPTIONS -- "
fi

CMD+="${args[*]}"
echo "Running: $CMD"
srun $CMD