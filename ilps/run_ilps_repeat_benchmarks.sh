#!/bin/bash
#
# Repeat ILPS-guided benchmarks N times using existing ILP solutions.
#
# This script assumes that:
#   1. The configuration JSON already exists.
#   2. Profiling & regression have been performed → a regression JSON exists.
#   3. ILP solutions have been generated → a solutions JSON exists.
#
# It will therefore skip all heavy preparation steps and simply submit the
# benchmark jobs multiple times to gather independent timing samples.
#
# Usage:
#   ./ilps/run_ilps_repeat_benchmarks.sh \
#       --config CONFIG_FILE \
#       [--solutions-file FILE] \
#       [--ngpus N] [--runs N] \
#       [--account NAME] [--constraint NAME]
#
# Example (run 10 identical benchmarks):
#   ./ilps/run_ilps_repeat_benchmarks.sh \
#       --config ilps/configs/default.json \
#       --solutions-file results/ilps-solutions/default.json \
#       --ngpus 4 --runs 10
#
set -euo pipefail

# --------------------------- Default parameters --------------------------- #
CONFIG_FILE=""
SOLUTIONS_FILE=""
NGPUS=4
NRUNS=5
SLURM_ACCOUNT=""
SLURM_CONSTRAINT=""
SLURM_MORE_OPTS=""

# --------------------------- Parse CLI arguments -------------------------- #
while [[ $# -gt 0 ]]; do
    case $1 in
    --config)
        CONFIG_FILE="$2"
        shift 2
        ;;
    --solutions-file)
        SOLUTIONS_FILE="$2"
        shift 2
        ;;
    --ngpus)
        NGPUS="$2"
        shift 2
        ;;
    --runs)
        NRUNS="$2"
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
    --slurm-opts)
        SLURM_MORE_OPTS="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter: $1" >&2
        exit 1
        ;;
    esac
done

# --------------------------- Sanity checks ------------------------------- #
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config is required" >&2
    exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi
CONFIG_NAME=$(basename "$CONFIG_FILE" .json)

if [[ -z "$SOLUTIONS_FILE" ]]; then
    SOLUTIONS_FILE="results/ilps-solutions/${CONFIG_NAME}.json"
fi
if [[ ! -f "$SOLUTIONS_FILE" ]]; then
    echo "Solutions file not found: $SOLUTIONS_FILE" >&2
    exit 1
fi

mkdir -p results logs

# --------------------------- Build SLURM options ------------------------- #
SLURM_OPTS="${SLURM_MORE_OPTS} "
[[ -n "$SLURM_ACCOUNT" ]] && SLURM_OPTS+="-A $SLURM_ACCOUNT "
[[ -n "$SLURM_CONSTRAINT" ]] && SLURM_OPTS+="-C $SLURM_CONSTRAINT "

# --------------------------- Submit benchmark runs ----------------------- #
echo "Submitting $NRUNS benchmark repetitions …"
for run_idx in $(seq 1 "$NRUNS"); do
    OUTPUT_FILE="results/bench-ilps-${CONFIG_NAME}-run${run_idx}.json"
    JOB_SCRIPT="results/run_benchmarks_${CONFIG_NAME}_run${run_idx}.sh"

    rm -f "$OUTPUT_FILE" "$JOB_SCRIPT"

    python ilps/generate_benchmark_jobs.py \
        --solutions-file "$SOLUTIONS_FILE" \
        --config-file "$CONFIG_FILE" \
        --output-file "$OUTPUT_FILE" \
        --ngpus "$NGPUS" \
        --slurm-opts "$SLURM_OPTS" \
        --output-script "$JOB_SCRIPT"

    echo "→ [$run_idx/$NRUNS] sbatch script generated: $JOB_SCRIPT"
    bash "$JOB_SCRIPT"
done

echo "All $NRUNS runs submitted. Aggregate afterwards with:"
echo "  python ilps/aggregate_benchmark_runs.py --pattern 'results/bench-ilps-${CONFIG_NAME}-run*.json' --output-file 'results/bench-ilps-${CONFIG_NAME}-stats.json'"
