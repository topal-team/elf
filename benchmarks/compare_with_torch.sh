# Pass account and partition information as arguments to this script as you would to sbatch

run_id=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
for pp in 4 8 16; do
    for schedule in 1f1b megatron zbh1 zbv; do
        sbatch --job-name ${schedule}-${pp} --output logs/${schedule}-${pp}.out --error logs/${schedule}-${pp}.err --gpus $pp --time 01:00:00 "$@" jz.sh \
            benchmarks/compare_with_torch.py --pp $pp --schedule $schedule --run-id $run_id --niters 30 --mb-size 1 --config-file models/configs/ppcm9-6.json
    done
done
