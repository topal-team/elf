#!/bin/bash

export OMP_NUM_THREADS=1
# See Jean-Zay doc on nsight systems
export TMPDIR=/gpfsscratch/rech/hyb/uak69sb/


nsight="nsys profile -w true -t nvtx -s none --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true --trace-fork-before-exec true -f true -o /mnt/profiling/%p.nsys"
run="torchrun --nnodes 1 --nproc-per-node 4 --standalone --no-python -- ${nsight} python /mnt/${@:1}"
# run="python /mnt/${@:1}"

echo Executing : "${run}"
singularity exec --nv --bind ${PWD}:/mnt,${TMPDIR}:${TMPDIR} ${SINGULARITY_ALLOWED_DIR}/pipe.sif \
     bash -c "${run}"
