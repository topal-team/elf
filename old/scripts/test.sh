#!/bin/bash -l

pytest -m single . $*
if [ -z "$(nvidia-smi -L 2>/dev/null)" ]; then
    echo "No GPUs found, skipping multi-GPU tests"
    exit 0
fi

n_gpus=$(nvidia-smi -L | wc -l)
if [ "$n_gpus" -le 1 ]; then
    echo "Not enough GPUs available (found $n_gpus, need 2), skipping multi-GPU tests"
    exit 0
fi

torchrun --nproc-per-node $n_gpus -m pytest -m multi . $*
