for schedule in afab 1f1b hanayo full_remat; do
	for partitioner in naive constrained metis dagP; do
		echo "Running schedule $schedule with partitioner $partitioner"
		torchrun --nproc_per_node=4 benchmarks/bench_gpt.py --schedule $schedule --partitioner $partitioner --model large --batch_size 32
	done
done
