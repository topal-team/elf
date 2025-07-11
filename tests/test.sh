#!/bin/bash

pytest

n_gpus=$(nvidia-smi -L 2> /dev/null | wc -l)
if [ "$n_gpus" -lt 2 ]; then
    echo "Need at least 2 GPUs for multi-GPU tests, found $n_gpus"
    exit 0
fi

./tests/distributed/distributed_tests.sh

# Assert the most basic script runs
echo "Checking that a basic example runs"
torchrun --nproc-per-node $n_gpus examples/basic.py
torchrun --nproc-per-node $n_gpus examples/basic_no_partition.py

echo "Checking for correctness"
torchrun --nproc-per-node $n_gpus -- tests/distributed/dp.py -dp 1 -pp $n_gpus
