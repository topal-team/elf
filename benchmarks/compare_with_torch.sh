run_id=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
for pp in 4 8 16 32 64; do
    sbatch --gpus $pp jz.sh benchmarks/compare_with_torch.py --pp $pp --run_id $run_id
done