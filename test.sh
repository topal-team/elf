#!/bin/bash

singularity exec --nv --bind $(pwd):/mnt \
	    $SINGULARITY_ALLOWED_DIR/pipe.sif \
	    bash -c \
	    "
	    
	    torchrun --nproc-per-node 4 --standalone --no-python -- \
	    python -m pytest /mnt -k test_block_multi -s
	    "


