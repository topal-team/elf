#!/bin/bash

# Default values
BASE_CONFIG_FILE=""
NGPUS=4
MIN_SEQLEN=128
MAX_SEQLEN=2048
STEP=128
NBLOCKS=16
SCHEDULER="zbh2"
MEMGPU=28000
SDP_BACKEND="None"
PRECISION="fp32"
SLURM_ACCOUNT=""
SLURM_CONSTRAINT=""

# Parse named parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            BASE_CONFIG_FILE="$2"
            shift 2
            ;;
        --ngpus)
            NGPUS="$2"
            shift 2
            ;;
        --min-seqlen)
            MIN_SEQLEN="$2"
            shift 2
            ;;
        --max-seqlen)
            MAX_SEQLEN="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --nblocks)
            NBLOCKS="$2"
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
        --sdp-backend)
            SDP_BACKEND="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --constraint)
            SLURM_CONSTRAINT="$2"
            shift 2
            ;;
        --account)
            SLURM_ACCOUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Ensure config file is provided
if [ -z "$BASE_CONFIG_FILE" ]; then
    echo "Error: No config file provided"
    echo "Usage: $0 --config <base_config_file> [--ngpus N] [--min-seqlen N] [--max-seqlen N] [--step N] [--nblocks N] [--scheduler NAME] [--memgpu N] [--account NAME] [--constraint NAME]"
    exit 1
fi

# Check if the base config file exists
if [ ! -f "$BASE_CONFIG_FILE" ]; then
    echo "Error: Base config file '$BASE_CONFIG_FILE' not found"
    exit 1
fi

# Build SLURM options
SLURM_OPTS=""
if [ ! -z "$SLURM_ACCOUNT" ]; then
    SLURM_OPTS+="-A $SLURM_ACCOUNT "
fi
if [ ! -z "$SLURM_CONSTRAINT" ]; then
    SLURM_OPTS+="-C $SLURM_CONSTRAINT "
fi

BASE_CONFIG_NAME=$(basename $BASE_CONFIG_FILE .json)
RESULTS_DIR="results/seqlen_benchmark"

# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Generate configs with different sequence lengths
echo "Generating configs with different sequence lengths..."
python ilps/generate_seqlen_configs.py --base_config $BASE_CONFIG_FILE --min_seqlen $MIN_SEQLEN --max_seqlen $MAX_SEQLEN --step $STEP

# Find all generated configs
CONFIG_DIR=$(dirname $BASE_CONFIG_FILE)
SEQLEN_CONFIGS_DIR="$CONFIG_DIR/seqlen_configs"
SEQLEN_CONFIGS=$(find $SEQLEN_CONFIGS_DIR -name "${BASE_CONFIG_NAME}_seqlen_*.json" -type f | sort)

# Check if we have configs
if [ -z "$SEQLEN_CONFIGS" ]; then
    echo "Error: No sequence length configs generated"
    exit 1
fi

echo "Found $(echo "$SEQLEN_CONFIGS" | wc -l) sequence length configs to benchmark"

# Process each config
for CONFIG_FILE in $SEQLEN_CONFIGS; do
    CONFIG_NAME=$(basename $CONFIG_FILE .json)
    SEQLEN=$(echo $CONFIG_NAME | grep -o "seqlen_[0-9]*" | cut -d_ -f2)
    sbatch $SLURM_OPTS --gpus 2 --time=01:00:00 --job-name=seqlen-${SEQLEN} --output=logs/seqlen-${SEQLEN}.out --error=logs/seqlen-${SEQLEN}.err jz.sh --no-python ilps/run_one_ilps_seqlen_benchmark.sh $CONFIG_FILE $RESULTS_DIR $NGPUS $NBLOCKS $SCHEDULER $MEMGPU $NBLOCKS $SLURM_OPTS $SDP_BACKEND $PRECISION
done

echo "All benchmark jobs enqueued, commands are appended to file $RESULTS_DIR/run_benchmarks_${BASE_CONFIG_NAME}.sh"
echo "In order to merge the results afterwards, run:"
echo "python ilps/merge_seqlen_results.py --results_dir $RESULTS_DIR --output_file $RESULTS_DIR/summary.json"