#!/bin/bash


singularity exec --nv --bind \
	    $(pwd):$(pwd),$SCRATCH:/data,$DSDIR/HuggingFace_Models:/models \
	    $SINGULARITY_ALLOWED_DIR/pipe.sif \
	    torchrun --standalone --nproc-per-node 4 -- \
	    scripts/train_llama.py $*
