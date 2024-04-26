#!/bin/bash

singularity exec --nv --bind $(pwd):/mnt \
	    $SINGULARITY_ALLOWED_DIR/pipe.sif \
	    bash -c \
	    "
	    python -m pytest /mnt -m single -q ;
	    torchrun --nproc-per-node 2 --standalone --no-python -- \
	    python -m pytest /mnt -m multi -q
	    "


