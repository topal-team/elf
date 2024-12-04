#!/bin/bash

pytest -m single . $*

n_gpus=$(nvidia-smi -L 2> /dev/null | wc -l)
if [ "$n_gpus" -lt 2 ]; then
    echo "Need at least 2 GPUs for multi-GPU tests, found $n_gpus"
    exit 0
fi

torchrun --nproc-per-node $n_gpus -m pytest -m multi . $*

echo "Checking that a basic example runs"
torchrun --nproc-per-node $n_gpus examples/basic.py # Assert the most basic script runs

echo "Checking for correctness"
torchrun --nproc-per-node $n_gpus -- tests/dp.py -dp 1 -pp 4
