#!/bin/bash

CONFIG_FILE=$1
CONFIG_NAME=$(basename $CONFIG_FILE .json)
SEQLEN=$(echo $CONFIG_NAME | grep -o "seqlen_[0-9]*" | cut -d_ -f2)
BASE_CONFIG_NAME=$(echo $CONFIG_NAME | sed 's/_seqlen_[0-9]*$//')
RESULTS_DIR=$2
NGPUS=$3
SCHEDULER=$4
MEMGPU=$5
SLURM_OPTS="$6 $7 $8 $9"

# Run profiling
python -u ilps/profiling.py --config $CONFIG_FILE --output $RESULTS_DIR/profiling_${CONFIG_NAME}.json

# Run regression
echo "Running regression..."
python ilps/regression.py --input-file $RESULTS_DIR/profiling_${CONFIG_NAME}.json --config-file $CONFIG_FILE --nstages $NGPUS

# Run communication profiling
echo "Running communication profiling..."
torchrun --nproc-per-node=2 ilps/profiling-comms.py --config $CONFIG_FILE

# Clean up any existing solutions file
if [[ -f $RESULTS_DIR/ilps-solutions_${CONFIG_NAME}.json ]]; then
    echo "!! Warning: $RESULTS_DIR/ilps-solutions_${CONFIG_NAME}.json already exists, deleting it."
    rm -f $RESULTS_DIR/ilps-solutions_${CONFIG_NAME}.json
fi

# Generate solutions
echo "Generating solutions..."
python pipeline-ilps/runall.py --config $CONFIG_FILE --output $RESULTS_DIR/ilps-solutions_${CONFIG_NAME}.json --processors $NGPUS --scheduler $SCHEDULER --mem $MEMGPU
python pipeline-ilps/generate_baselines.py --config $CONFIG_FILE --output $RESULTS_DIR/ilps-solutions_${CONFIG_NAME}.json --processors $NGPUS --scheduler $SCHEDULER

# Add default SLURM options if none provided
if [ -z "$SLURM_OPTS" ]; then
    SLURM_OPTS="-A gdh@h100 -C h100"
fi

# Generate benchmark jobs
python ilps/generate_benchmark_jobs.py --solutions-file $RESULTS_DIR/ilps-solutions_${CONFIG_NAME}.json --config-file $CONFIG_FILE --output-file $RESULTS_DIR/bench-ilps-${CONFIG_NAME}.json --ngpus $NGPUS --slurm-opts "$SLURM_OPTS" --output-script $RESULTS_DIR/run_benchmarks_${BASE_CONFIG_NAME}.sh

# Run the benchmark
# echo "Running benchmark for sequence length $SEQLEN..."
# torchrun --nproc-per-node=$NGPUS benchmarks/ilps_guided_benchmark.py --restart --config_file $CONFIG_FILE --solution_file $RESULTS_DIR/ilps-solutions_${CONFIG_NAME}.json --output_file $RESULTS_DIR/bench-ilps-${CONFIG_NAME}.json --base $SCHEDULER --n $NBLOCKS

# echo "Completed benchmark for sequence length $SEQLEN"