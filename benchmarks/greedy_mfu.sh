#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# greedy_mfu.sh
# -----------------------------------------------------------------------------
# End-to-end automation script that:
#   1. Profiles a model (compute + communication) to populate its config JSON.
#   2. Generates greedy schedules for different GPU counts.
#   3. Benchmarks each schedule with the distributed training driver.
#
# The benchmarks are dispatched in parallel via ``sbatch`` so the individual
# runs do not block each other.  You can control most runtime parameters
# through environment variables (see below).
# -----------------------------------------------------------------------------
# Usage
#   ./benchmarks/greedy_mfu.sh [--skip-profile] <CONFIG_JSON> [GPU_COUNTS ...]
#
# Arguments
#   --skip-profile   Skip compute & communication profiling (uses existing values).
#   CONFIG_JSON      Path to a *base* JSON configuration describing the model
#                    and ILP parameters (will be updated in-place).
#   GPU_COUNTS       Space-separated list of pipeline parallel degrees (defaults
#                    to 4 8 16 32).
#
# Environment variables (with defaults):
#   BENCH_ITERS   Training iterations per benchmark (10, overridable).
#   SBATCH_OPTS   Extra flags forwarded directly to every sbatch invocation.
#   RESULTS_DIR   Output directory (results/greedy_mfu).
#   GPU_MEM       Amount of memory per GPU (defaults to 72G).
# -----------------------------------------------------------------------------

set -euo pipefail

# Parse optional flags
SKIP_PROFILE=0
while [[ $# -gt 0 && $1 == --* ]]; do
    case $1 in
    --skip-profile)
        SKIP_PROFILE=1
        shift
        ;;
    --memgpu)
        GPU_MEM=$2
        shift 2
        ;;
    *)
        echo "Unknown option: $1" >&2
        exit 1
        ;;
    esac
done

if [[ $# -lt 1 ]]; then
    cat <<EOF
Usage: $0 <CONFIG_JSON> [GPU_COUNTS ...]

Run greedy MFU pipeline:
  * Update <CONFIG_JSON> with compute & comm profiling
  * Generate greedy schedules for each GPU count
  * Launch distributed benchmarks via sbatch
EOF
    exit 1
fi

# -----------------------------------------------------------------------------
# User inputs
# -----------------------------------------------------------------------------
# Positional arguments after flag parsing
CONFIG_JSON=$(realpath "$1")
shift || true

# Remaining arguments form the list of GPU counts; default to 4 8 16 32 if none given
if [[ $# -gt 0 ]]; then
    GPU_LIST=("$@")
else
    GPU_LIST=(4 8 16 32)
fi

# -----------------------------------------------------------------------------
# Customisable knobs (via env-vars)
# -----------------------------------------------------------------------------
BENCH_ITERS=${BENCH_ITERS:-10}
# Default to GPU partition unless user overrides
SBATCH_OPTS=${SBATCH_OPTS:-}
RESULTS_DIR=${RESULTS_DIR:-results/greedy_mfu}
GPU_MEM=${GPU_MEM:-72}
mkdir -p "$RESULTS_DIR"

# Copy config in a temp workspace to avoid clobbering original
WORK_CFG="$RESULTS_DIR/$(basename "$CONFIG_JSON")"
cp "$CONFIG_JSON" "$WORK_CFG"

# -----------------------------------------------------------------------------
# 1. Profiling (compute & comm) – dispatched to GPU nodes via sbatch (unless skipped)
# -----------------------------------------------------------------------------

if [[ $SKIP_PROFILE -eq 0 ]]; then
    profile_out="$RESULTS_DIR/profiling_stats.json"

    # 1.1 Compute kernels profiling (single-GPU)
    echo "[1/3-a] Submitting compute-profiling job…"
    job_name_prof="profile_compute"
    log_prof="$RESULTS_DIR/${job_name_prof}.log"

    prof_jobid=$(
        sbatch $SBATCH_OPTS \
            --wait \
            --job-name "$job_name_prof" \
            --gpus 1 \
            --output "$log_prof" \
            jz.sh --no-multi -u ilps/profiling.py \
            --config-file "$WORK_CFG" \
            --output "$profile_out"
    )

    echo "    → Compute profiling job $prof_jobid finished."

    source venv/bin/activate
    python ilps/regression.py --input-file $profile_out --config-file $WORK_CFG --output-file $WORK_CFG --nstages 1

    # 1.2 Communication profiling (needs ≥2 GPUs)
    echo "[1/3-b] Submitting communication-profiling job…"
    job_name_comm="profile_comm"
    log_comm="$RESULTS_DIR/${job_name_comm}.log"

    comm_jobid=$(
        sbatch $SBATCH_OPTS \
            --wait \
            --job-name "$job_name_comm" \
            --gpus 2 \
            --output "$log_comm" \
            jz.sh --no-python torchrun --standalone --nproc-per-node=2 ilps/profiling-comms.py \
            --config-file "$WORK_CFG"
    )

    echo "    → Communication profiling job $comm_jobid finished."
else
    echo "[1/3] Skipping profiling as requested (--skip-profile)"
fi

# -----------------------------------------------------------------------------
# 2. Generate greedy schedules for each GPU count
# -----------------------------------------------------------------------------
for GPUS in "${GPU_LIST[@]}"; do
    sched_out="$RESULTS_DIR/schedule_${GPUS}.json"
    echo "[2/3] Generating greedy schedule for ${GPUS} GPUs → $sched_out"
    python pipeline-ilps/greedy.py \
        --config "$WORK_CFG" \
        --processors "$GPUS" \
        --nmb $((3 * GPUS)) \
        -s lookahead \
        --output "$sched_out" \
        --memgpu "$GPU_MEM"
done

# -----------------------------------------------------------------------------
# 3. Launch benchmarks (one sbatch per GPU count) using jz.sh
# -----------------------------------------------------------------------------

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for GPUS in "${GPU_LIST[@]}"; do
    sched_out="$RESULTS_DIR/schedule_${GPUS}.json"
    job_name="greedy_${GPUS}g"
    out_file="$LOG_DIR/${job_name}.out"
    err_file="$LOG_DIR/${job_name}.err"

    echo "[3/3] Submitting benchmark for ${GPUS} GPUs (logs: ${out_file}, ${err_file})"
    sbatch $SBATCH_OPTS \
        --job-name "$job_name" \
        --gpus $GPUS \
        --output "$out_file" \
        --error "$err_file" \
        jz.sh benchmarks/run_transformer.py \
        --scheduler file \
        --schedule-file "$sched_out" \
        --nmb $((3 * GPUS)) \
        --niters "$BENCH_ITERS" \
        --log info \
        --config-file "$WORK_CFG" \
        --partitioner constrained
done

echo "All jobs submitted.  Monitor progress with: tail -f $LOG_DIR/*.out"
