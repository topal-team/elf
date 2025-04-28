run_id=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
for pp in 4 8 16 32 64; do
    for schedule in 1f1b zbh1 afab; do
        sbatch --job-name ${schedule}-${pp} --output logs/${schedule}-${pp}.out --error logs/${schedule}-${pp}.err --gpus $pp jz.sh benchmarks/compare_with_torch.py --pp $pp --schedule $schedule --run_id $run_id
    done
done