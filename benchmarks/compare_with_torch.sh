run_id=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
for pp in 4 8 16 24 32 48 64; do
    for schedule in afab 1f1b megatron zbh1 zbv; do
        sbatch --job-name ${schedule}-${pp} --output logs/${schedule}-${pp}.out --error logs/${schedule}-${pp}.err --gpus $pp --time 01:00:00 jz.sh benchmarks/compare_with_torch.py --pp $pp --schedule $schedule --run_id $run_id --niters 30
    done
done