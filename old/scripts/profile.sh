#!/bin/bash

export OMP_NUM_THREADS=1
# See Jean-Zay doc on nsight systems
export TMPDIR=/gpfsscratch/rech/hyb/uak69sb/

if [ ! -d "profiling" ]; then
    mkdir profiling
else
    rm -f profiling/* ;
fi

nsight="nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o profiling/%p.nsys"
run="torchrun --nnodes 1 --nproc-per-node 4 --standalone --no-python -- ${nsight} python ${@:1}"
# run="python /mnt/${@:1}"

echo Executing : "${run}"
singularity exec --nv --bind ${PWD}:${PWD},${TMPDIR}:${TMPDIR} ${SINGULARITY_ALLOWED_DIR}/pipe.sif \
     bash -c "${run}"
