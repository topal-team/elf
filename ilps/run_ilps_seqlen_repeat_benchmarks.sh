#!/bin/bash
#
# Repeat ILPS benchmarks for a range of sequence lengths, N times each, using
# pre-computed ILP solutions.
#
# This script is analogous to `run_ilps_seqlen_benchmark.sh`, but instead of
# running the heavy profiling / regression / ILP solving steps every time, it
# expects that those artefacts already exist for each generated sequence length
# configuration, and focuses on repeating the benchmark execution to collect
# multiple samples.
#
# Usage:
#   ./ilps/run_ilps_seqlen_repeat_benchmarks.sh \
#       --config BASE_CONFIG.json \
#       [--ngpus N] [--min-seqlen N] [--max-seqlen N] [--step N] \
#       [--runs N] [--scheduler NAME] [--memgpu MB] \
#       [--account NAME] [--constraint NAME]
#
# The script will:
#   1. Generate per-sequence-length config JSONs via generate_seqlen_configs.py
#      (unless they already exist).
#   2. For each config, locate its ILP solutions JSON under
#        results/seqlen_benchmarks/ilps-solutions_<config_name>.json
#      and call `run_ilps_repeat_benchmarks.sh` to submit the benchmark N times.
#
set -euo pipefail

# --------------------------- Default parameters --------------------------- #
BASE_CONFIG_FILE=""
NGPUS=4
MIN_SEQLEN=128
MAX_SEQLEN=2048
STEP=128
NRUNS=5
SCHEDULER="zbh2"
MEMGPU=28000
SLURM_ACCOUNT=""
SLURM_CONSTRAINT=""

# --------------------------- Parse CLI arguments -------------------------- #
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
    --runs)
        NRUNS="$2"
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
    *)
        echo "Unknown parameter: $1" >&2
        exit 1
        ;;
    esac
done

# --------------------------- Sanity checks ------------------------------- #
if [[ -z "$BASE_CONFIG_FILE" ]]; then
    echo "Error: --config BASE_CONFIG.json is required" >&2
    exit 1
fi
if [[ ! -f "$BASE_CONFIG_FILE" ]]; then
    echo "Base config file not found: $BASE_CONFIG_FILE" >&2
    exit 1
fi

BASE_CONFIG_NAME=$(basename "$BASE_CONFIG_FILE" .json)
RESULTS_DIR="results/seqlen_benchmarks"
mkdir -p "$RESULTS_DIR" logs

# --------------------------- SLURM opts ---------------------------------- #
SLURM_OPTS=""
[[ -n "$SLURM_ACCOUNT" ]] && SLURM_OPTS+="-A $SLURM_ACCOUNT "
[[ -n "$SLURM_CONSTRAINT" ]] && SLURM_OPTS+="-C $SLURM_CONSTRAINT "

# --------------------------- Generate configs ---------------------------- #
CONFIG_DIR=$(dirname "$BASE_CONFIG_FILE")
SEQLEN_CONFIGS_DIR="$CONFIG_DIR/seqlen_configs"
if [[ ! -d "$SEQLEN_CONFIGS_DIR" ]]; then
    mkdir -p "$SEQLEN_CONFIGS_DIR"
fi

echo "[1/3] Ensuring sequence-length specific configs exist …"
python ilps/generate_seqlen_configs.py \
    --base-config "$BASE_CONFIG_FILE" \
    --min-seqlen "$MIN_SEQLEN" --max-seqlen "$MAX_SEQLEN" --step "$STEP"

SEQLEN_CONFIGS=$(find "$SEQLEN_CONFIGS_DIR" -name "${BASE_CONFIG_NAME}_seqlen_*.json" -type f | sort)
if [[ -z "$SEQLEN_CONFIGS" ]]; then
    echo "No sequence length configs generated" >&2
    exit 1
fi

NUM_CFGS=$(echo "$SEQLEN_CONFIGS" | wc -l)
echo "Found $NUM_CFGS sequence length configs. Submitting $NRUNS benchmark repeats each."

# --------------------------- Loop over configs --------------------------- #
for CONFIG_FILE in $SEQLEN_CONFIGS; do
    CONFIG_NAME=$(basename "$CONFIG_FILE" .json)
    SEQLEN=$(echo "$CONFIG_NAME" | grep -o "seqlen_[0-9]*" | cut -d_ -f2)

    SOL_FILE="$RESULTS_DIR/ilps-solutions_${CONFIG_NAME}.json"
    if [[ ! -f "$SOL_FILE" ]]; then
        echo "!! Warning: Solutions file $SOL_FILE not found. Please generate ILP solutions first (e.g., with run_one_ilps_seqlen_benchmark.sh). Skipping seq=$SEQLEN."
        continue
    fi

    echo "Submitting benchmarks for sequence length $SEQLEN …"
    ilps/run_ilps_repeat_benchmarks.sh \
        --config "$CONFIG_FILE" \
        --solutions-file "$SOL_FILE" \
        --ngpus "$NGPUS" \
        --runs "$NRUNS" \
        --account "$SLURM_ACCOUNT" \
        --constraint "$SLURM_CONSTRAINT" \
        --slurm-opts "$SLURM_OPTS"

done

echo "[Done] All repeat benchmark jobs submitted for sequence-length sweep."
