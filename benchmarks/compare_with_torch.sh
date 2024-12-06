for pp in 4 8 16 32; do
    sbatch --gpus $pp jz.sh benchmarks/compare_with_torch.py --pp $pp
done