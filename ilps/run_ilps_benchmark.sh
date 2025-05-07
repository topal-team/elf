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

# Check if a config file was provided
if [ $# -lt 1 ]; then
    echo "Error: No config file provided"
    echo "Usage: $0 <config_file> [ngpus] [min_blocks] [max_blocks] [step] [scheduler]"
    exit 1
fi

# Default values
CONFIG_FILE=""
NGPUS=4
MIN_BLOCKS=4
MAX_BLOCKS=16
STEP=4
SCHEDULER="zbh2"
MEMGPU=24000

# Parse named parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --ngpus)
            NGPUS="$2"
            shift 2
            ;;
        --min-blocks)
            MIN_BLOCKS="$2"
            shift 2
            ;;
        --max-blocks)
            MAX_BLOCKS="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --memgpu)
            MEMGPU="$2"
            shift 2
            ;;
        *)
            # For backwards compatibility, treat first unnamed arg as config file
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"
            fi
            shift
            ;;
    esac
done

# Ensure config file is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file provided"
    echo "Usage: $0 --config <config_file> [--ngpus N] [--min-blocks N] [--max-blocks N] [--step N] [--scheduler NAME] [--memgpu N]"
    exit 1
fi

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

CONFIG_NAME=$(basename $CONFIG_FILE .json)

mkdir -p results/profiling
mkdir -p results/regression

python ilps/profiling.py --config $CONFIG_FILE --output results/profiling/$CONFIG_NAME.json -i 30

python ilps/regression.py --input_file results/profiling/$CONFIG_NAME.json --config_file $CONFIG_FILE --output_file results/regression/$CONFIG_NAME.json

torchrun --nproc-per-node=2 ilps/profiling-comms.py --config results/regression/$CONFIG_NAME.json

if [[ -f results/ilps-solutions/$CONFIG_NAME.json ]]; then
    echo "!! Warning: results/ilps-solutions/$CONFIG_NAME.json already exists, deleting it."
    rm -f results/ilps-solutions/$CONFIG_NAME.json
fi

for nblocks in $(seq $MIN_BLOCKS $STEP $MAX_BLOCKS) ; do
    python ~/pipeline-ilps/runall.py --config results/regression/$CONFIG_NAME.json --nblocks $nblocks --output results/ilps-solutions/$CONFIG_NAME.json --processors $NGPUS --time-limit 120 --scheduler $SCHEDULER --mem $MEMGPU
done

echo "Solutions generated. To run the benchmark, run:"
echo "torchrun --nproc-per-node=$NGPUS benchmarks/ilps_guided_benchmark.py --restart --config_file $CONFIG_FILE --solution_file results/ilps-solutions/$CONFIG_NAME.json --output_file results/bench-ilps-$CONFIG_NAME.json --base $SCHEDULER"

# Check if there are enough GPUs on the current node
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $AVAILABLE_GPUS -ge $NGPUS ]; then
    echo "Found $AVAILABLE_GPUS GPUs, which is enough to run the benchmark with $NGPUS GPUs."
    echo "Executing benchmark..."
    torchrun --nproc-per-node=$NGPUS benchmarks/ilps_guided_benchmark.py --restart --config_file $CONFIG_FILE --solution_file results/ilps-solutions/$CONFIG_NAME.json --output_file results/bench-ilps-$CONFIG_NAME.json --base $SCHEDULER
    echo "Benchmark completed successfully."
else
    echo "Not enough GPUs available. Found $AVAILABLE_GPUS, but need $NGPUS."
    echo "Skipping benchmark execution."
fi
