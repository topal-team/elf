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
#   ./ilps/run_ilps_benchmark.sh --config CONFIG_FILE [--ngpus N] [--min-blocks N] [--max-blocks N] [--step N] [--scheduler NAME] [--memgpu N] [--account NAME] [--constraint NAME] [--regression-file FILE]
#
# Arguments:
#   --config: Path to the configuration file with model hyperparameters
#   --ngpus: Number of GPUs to use for benchmarking (default: 4)
#   --min-blocks: Minimum number of transformer blocks to test (default: 4)
#   --max-blocks: Maximum number of transformer blocks to test (default: 16)
#   --step: Step size for block count increments (default: 4)
#   --scheduler: Base scheduler type (default: zbh2)
#   --memgpu: GPU memory limit in MB (default: 24000)
#   --account: SLURM account name
#   --constraint: SLURM constraint (e.g., h100, v100-32g, ..)
#   --regression-file: Path to existing regression file (if provided, skips profiling steps)
#
# Example:
#   ./ilps/run_ilps_benchmark.sh --config ilps/configs/default.json --ngpus 4 --min-blocks 4 --max-blocks 16 --step 4

# Default values
CONFIG_FILE=""
NGPUS=4
MIN_BLOCKS=4
MAX_BLOCKS=16
STEP=4
SCHEDULER="zbh2"
MEMGPU=24000
SLURM_ACCOUNT=""
SLURM_CONSTRAINT=""
REGRESSION_FILE=""
SDP_BACKEND=""

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
        --account)
            SLURM_ACCOUNT="$2"
            shift 2
            ;;
        --constraint)
            SLURM_CONSTRAINT="$2"
            shift 2
            ;;
        --regression-file)
            REGRESSION_FILE="$2"
            shift 2
            ;;
        --sdp-backend)
            SDP_BACKEND="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Ensure config file is provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file provided"
    echo "Usage: $0 --config <config_file> [--ngpus N] [--min-blocks N] [--max-blocks N] [--step N] [--scheduler NAME] [--memgpu N] [--account NAME] [--constraint NAME] [--regression-file FILE]"
    exit 1
fi

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Check if regression file exists if provided
if [ ! -z "$REGRESSION_FILE" ] && [ ! -f "$REGRESSION_FILE" ]; then
    echo "Error: Regression file '$REGRESSION_FILE' not found"
    exit 1
fi

CONFIG_NAME=$(basename $CONFIG_FILE .json)

mkdir -p results/profiling
mkdir -p results/regression
mkdir -p results/ilps-solutions
mkdir -p results/benchmarks

# Function to setup environment based on GPU type
setup_gpu_env() {
    export OMP_NUM_THREADS=4
    
    if [[ "$SLURM_JOB_PARTITION" == "gpu_p5" ]]; then
        module load arch/a100
        source ~/scratch/venv-a100/bin/activate
        echo "Activated A100 virtual environment."
    elif [[ "$SLURM_JOB_PARTITION" == "gpu_p6" ]]; then
        module load arch/h100 cuda/12.8.0
        source ~/work/venv-h100/bin/activate 
        echo "Activated H100 virtual environment."
    else
        source ~/elf-dev/venv/bin/activate
        echo "Activated default virtual environment."
    fi
    python -c "import torch; print('Torch version:', torch.__version__)"
}

# Build SLURM options
SLURM_OPTS=""
if [ ! -z "$SLURM_ACCOUNT" ]; then
    SLURM_OPTS+="-A $SLURM_ACCOUNT "
fi
if [ ! -z "$SLURM_CONSTRAINT" ]; then
    SLURM_OPTS+="-C $SLURM_CONSTRAINT "
fi

# Run GPU-intensive profiling steps only if no regression file is provided
if [ -z "$REGRESSION_FILE" ]; then
    echo "No regression file provided. Running profiling steps..."
    srun --time=00:30:00 $SLURM_OPTS --gpus=1 bash -c "
        $(declare -f setup_gpu_env)
        setup_gpu_env
        python -u ilps/profiling.py --config $CONFIG_FILE --output results/profiling/$CONFIG_NAME.json -i 30 --sdp-backend $SDP_BACKEND
        python ilps/regression.py --input_file results/profiling/$CONFIG_NAME.json --config_file $CONFIG_FILE --output_file results/regression/$CONFIG_NAME.json
    "
    srun --time=00:10:00 $SLURM_OPTS --gpus=2 --ntasks=1 bash -c "
        $(declare -f setup_gpu_env)
        setup_gpu_env
        torchrun --nproc-per-node=2 ilps/profiling-comms.py --config results/regression/$CONFIG_NAME.json
    "
    REGRESSION_FILE="results/regression/$CONFIG_NAME.json"
else
    echo "Using provided regression file: $REGRESSION_FILE"
fi

# Run ILP solving on the front node (no GPU needed)
if [[ -f results/ilps-solutions/$CONFIG_NAME.json ]]; then
    echo "!! Warning: results/ilps-solutions/$CONFIG_NAME.json already exists, deleting it."
    rm -f results/ilps-solutions/$CONFIG_NAME.json
fi

source ~/elf-dev/venv/bin/activate

echo "Running ILP solving on front node..."
for nblocks in $(seq $MIN_BLOCKS $STEP $MAX_BLOCKS) ; do
    python pipeline-ilps/runall.py --config $REGRESSION_FILE \
                --nblocks $nblocks --output results/ilps-solutions/$CONFIG_NAME.json \
                --processors $NGPUS --time-limit 600 --scheduler $SCHEDULER --mem $MEMGPU
    python pipeline-ilps/generate_baselines.py --config $REGRESSION_FILE \
                --output results/ilps-solutions/$CONFIG_NAME.json \
                --processors $NGPUS --nblocks $nblocks --scheduler $SCHEDULER
done

echo "Solutions generated. Generating benchmark jobs..."

# Generate benchmark job script
BENCHMARK_SCRIPT="results/benchmarks/run_benchmarks_${CONFIG_NAME}.sh"
if [ -f "$BENCHMARK_SCRIPT" ]; then
    echo "!! Warning: $BENCHMARK_SCRIPT already exists, deleting it."
    rm -f "$BENCHMARK_SCRIPT"
fi

python ilps/generate_benchmark_jobs.py \
    --solutions-file "results/ilps-solutions/$CONFIG_NAME.json" \
    --config-file "$CONFIG_FILE" \
    --output-file "results/bench-ilps-$CONFIG_NAME.json" \
    --base-scheduler "$SCHEDULER" \
    --ngpus "$NGPUS" \
    --slurm-opts "$SLURM_OPTS" \
    --output-script "$BENCHMARK_SCRIPT" \
    --sdp-backend "$SDP_BACKEND"

echo "Generated benchmark script: $BENCHMARK_SCRIPT"