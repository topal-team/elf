#!/bin/bash
# 
# ILPS Benchmark Runner Script
#
# Steps:
# 1. Profiling transformer model performance (profiling.py)
# 2. Regression analysis to extract timing coefficients (regression.py)
# 3. Profiling inter-GPU communication times (profiling-comms.py)
# 4. Running ILP solvers for different block counts (runall.py)
# 5. Benchmarking execution strategies (ilps_guided_benchmark.py)
#
# Usage:
#   ./ilps/run_ilps_benchmark.sh CONFIG_FILE [NGPUS] [MIN_BLOCKS] [MAX_BLOCKS] [STEP]
#
# Arguments:
#   CONFIG_FILE: Path to the configuration file with model hyperparameters
#   NGPUS: Number of GPUs to use for benchmarking (default: 4)
#   MIN_BLOCKS: Minimum number of transformer blocks to test (default: 4)
#   MAX_BLOCKS: Maximum number of transformer blocks to test (default: 16)
#   STEP: Step size for block count increments (default: 4)
#
# Example:
#   ./ilps/run_ilps_benchmark.sh ilps/configs/default.json 4 4 16 4

# Function to check command status and exit on failure
check_command() {
    local cmd_name=$1
    if [ $? -ne 0 ]; then
        echo "Error: $cmd_name failed. Stopping here."
        exit 1
    fi
}

# Check if a config file was provided
if [ $# -lt 1 ]; then
    echo "Error: No config file provided"
    echo "Usage: $0 <config_file> [ngpus] [min_blocks] [max_blocks] [step]"
    exit 1
fi

CONFIG_FILE=$1

NGPUS=4
# Default values for min and max blocks and step
MIN_BLOCKS=4
MAX_BLOCKS=16
STEP=4

if [ $# -gt 1 ]; then
    NGPUS=$2
fi

# Check if min_blocks is provided
if [ $# -gt 2 ]; then
    MIN_BLOCKS=$3
fi

# Check if max_blocks is provided
if [ $# -gt 3 ]; then
    MAX_BLOCKS=$4
fi

# Check if step is provided
if [ $# -gt 4 ]; then
    STEP=$5
fi

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

CONFIG_NAME=$(basename $CONFIG_FILE .json)

python ilps/profiling.py --config $CONFIG_FILE --output results/profiling/$CONFIG_NAME.json -i 30
check_command "profiling.py"

python ilps/regression.py --input_file results/profiling/$CONFIG_NAME.json --config_file $CONFIG_FILE
check_command "regression.py"

torchrun --nproc-per-node=2 ilps/profiling-comms.py --config $CONFIG_FILE
check_command "profiling-comms.py"

if [[ -f results/ilp-solutions/$CONFIG_NAME.json ]]; then
    echo "!! Warning: results/ilp-solutions/$CONFIG_NAME.json already exists, deleting it."
    rm -f results/ilp-solutions/$CONFIG_NAME.json
fi

for nblocks in $(seq $MIN_BLOCKS $STEP $MAX_BLOCKS) ; do
    python ilps/runall.py --config $CONFIG_FILE --nblocks $nblocks --output results/ilps-solutions/$CONFIG_NAME.json --processors $NGPUS --time-limit 120
    check_command "runall.py with nblocks=$nblocks"
done

echo "Solutions generated. To run the benchmark, run:"
echo "torchrun --nproc-per-node=$NGPUS benchmarks/ilps_guided_benchmark.py --restart --config_file $CONFIG_FILE --solution_file results/ilps-solutions/$CONFIG_NAME.json --output_file results/bench-ilps-$CONFIG_NAME.json"

# Check if there are enough GPUs on the current node
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $AVAILABLE_GPUS -ge $NGPUS ]; then
    echo "Found $AVAILABLE_GPUS GPUs, which is enough to run the benchmark with $NGPUS GPUs."
    echo "Executing benchmark..."
    torchrun --nproc-per-node=$NGPUS benchmarks/ilps_guided_benchmark.py --restart --config_file $CONFIG_FILE --solution_file results/ilps-solutions/$CONFIG_NAME.json --output_file results/bench-ilps-$CONFIG_NAME.json
    check_command "benchmark execution"
    echo "Benchmark completed successfully."
else
    echo "Not enough GPUs available. Found $AVAILABLE_GPUS, but need $NGPUS."
    echo "Skipping benchmark execution."
fi
