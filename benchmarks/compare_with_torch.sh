# Pass account and partition information as arguments to this script as you would to sbatch

run_id=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
for pp in 4 8 16 24 32 48 64; do
    for schedule in 1f1b; do
        sbatch --job-name ${schedule}-${pp} --output logs/${schedule}-${pp}.out --error logs/${schedule}-${pp}.err --gpus $pp --time 01:00:00 "$@" jz.sh \
            benchmarks/compare_with_torch.py --pp $pp --schedule $schedule --run_id $run_id --niters 30 --mb_size 2 --nblocks 64 --hidden_dim 2048 --seq_len 1024
    done
done