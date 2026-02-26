#!/bin/bash

# Submit benchmark suite jobs for different GPU scales
# Each scale runs as a separate SLURM job, then results are merged on front node

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
SLURM_ARGS="-C h100 -A hyb@h100 --exclusive"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

echo "Using config: $CONFIG"
echo "Submitting benchmark suite jobs..."

# Array to store job IDs
JOB_IDS=()

# Submit 4 GPU job
echo "Submitting 4 GPU benchmark..."
JOB_ID_4=$(sbatch $SLURM_ARGS --gpus=4 --parsable \
    --job-name=bench_4gpu \
	--output logs/bench_4gpu.out \
	--error logs/bench_4gpu.err \
    jz.sh benchmarks/suite/run.py --config "$CONFIG" --scales 4)
JOB_IDS+=($JOB_ID_4)
echo "  Job ID: $JOB_ID_4"

# Submit 8 GPU job
echo "Submitting 8 GPU benchmark..."
JOB_ID_8=$(sbatch $SLURM_ARGS --gpus=8 --parsable \
    --job-name=bench_8gpu \
    --output logs/bench_8gpu.out \
    --error logs/bench_8gpu.err \
    jz.sh benchmarks/suite/run.py --config "$CONFIG" --scales 8)
JOB_IDS+=($JOB_ID_8)
echo "  Job ID: $JOB_ID_8"

echo ""
echo "All jobs submitted!"
echo "  4 GPU job: $JOB_ID_4"
echo "  8 GPU job: $JOB_ID_8"
echo ""
echo "Waiting for jobs to complete..."
echo "(You can safely Ctrl+C - jobs will continue running)"
echo ""

# Wait for both jobs to complete
while true; do
    # Check if both jobs are done (not in queue anymore)
    JOB_4_STATUS=$(squeue -j $JOB_ID_4 -h -o "%T" 2>/dev/null || echo "COMPLETED")
    JOB_8_STATUS=$(squeue -j $JOB_ID_8 -h -o "%T" 2>/dev/null || echo "COMPLETED")

    if [[ "$JOB_4_STATUS" == "COMPLETED" ]] && [[ "$JOB_8_STATUS" == "COMPLETED" ]]; then
        break
    fi

    # Show status
    echo -ne "\r  4 GPU: $JOB_4_STATUS | 8 GPU: $JOB_8_STATUS  "
    sleep 10
done

echo ""
echo ""
echo "All jobs completed! Merging results..."
python "$SCRIPT_DIR/merge.py"

echo ""
echo "✅ Benchmark suite complete!"
echo "Results saved to: results/benchmark_suite_results.json"