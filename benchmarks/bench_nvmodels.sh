run_id=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
for model in "dit"; do
    for schedule in "1f1b" "zbh1" "zbh2"; do
        for partitioner in "naive" "metis"; do
            for ngpus in 8 16 32; do
                sbatch --gpus $ngpus jz.sh benchmarks/bench_nvmodels.py --schedule $schedule --model $model --partitioner $partitioner --run-id $run_id
            done
        done
    done
done
