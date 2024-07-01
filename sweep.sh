#!/bin/bash

#SBATCH --account=hyb@v100
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --exclusive

module load singularity

singularity exec --nv --bind $(pwd):$(pwd) ${SINGULARITY_ALLOWED_DIR}/pipe.sif ./agent.sh
