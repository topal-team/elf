#!/bin/bash


# Check if a config file was provided
if [ $# -lt 1 ]; then
    echo "Error: No base config file provided"
    echo "Usage: $0 <base_config_file> [ngpus] [min_seqlen] [max_seqlen] [step]"
    exit 1
fi

BASE_CONFIG_FILE=$1

# Default values
NGPUS=4
MIN_SEQLEN=128
MAX_SEQLEN=2048
STEP=128
NBLOCKS=16
SCHEDULER="zbh2"

# Parse arguments
if [ $# -gt 1 ]; then
    NGPUS=$2
fi

if [ $# -gt 2 ]; then
    MIN_SEQLEN=$3
fi

if [ $# -gt 3 ]; then
    MAX_SEQLEN=$4
fi

if [ $# -gt 4 ]; then
    STEP=$5
fi

if [ $# -gt 5 ]; then
    NBLOCKS=$6
fi

if [ $# -gt 6 ]; then
    SCHEDULER=$7
fi

# Check if the base config file exists
if [ ! -f "$BASE_CONFIG_FILE" ]; then
    echo "Error: Base config file '$BASE_CONFIG_FILE' not found"
    exit 1
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
    sbatch -C v100-32g --gpus 4 --time=01:00:00 --job-name=seqlen-${SEQLEN} --output=logs/seqlen-${SEQLEN}.out --error=logs/seqlen-${SEQLEN}.err ilps/run_one_ilps_seqlen_benchmark.sh $CONFIG_FILE $RESULTS_DIR $NGPUS $NBLOCKS $SCHEDULER
done

echo "All benchmark jobs submitted."
echo "In order to merge the results, run:"
echo "python ilps/merge_seqlen_results.py --results_dir $RESULTS_DIR --output_file $RESULTS_DIR/summary.json"