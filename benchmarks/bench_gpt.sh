for schedule in afab 1f1b hanayo full_remat zbh1 zbh2; do
    for partitioner in naive constrained metis dagP; do
        echo "Running schedule $schedule with partitioner $partitioner"
        sbatch --gpus 16 jz.sh benchmarks/bench_gpt.py --schedule $schedule --partitioner $partitioner --model xxl --batch_size 32 --dp 2 --pp 8
    done
done
