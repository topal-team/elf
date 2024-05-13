#!/bin/bash

singularity exec --nv --bind $(pwd):$(pwd) \
	    $SINGULARITY_ALLOWED_DIR/pipe.sif \
	    bash -c \
	    "
	    python -m pytest -m single .
	    torchrun --nproc-per-node 4 --standalone --no-python -- \
	    python -m pytest -s -m multi .
	    "


