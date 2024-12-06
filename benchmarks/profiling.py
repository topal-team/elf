import torch
import torch.distributed as dist

import os
import sys
import torch.cuda.profiler as profiler
from torch.cuda import cudart

sys.path.append(".")
from models.simple import SimpleTransformer
from elf import Pipeline

if __name__ == "__main__":
	# Initialize distributed training
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	ws = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	# Create model and sample data
	model = SimpleTransformer(4000, 2048, 8 * ws, 512)
	sample = model.get_sample(32).cuda()
	target = model.get_target(32).cuda()
	loss_fn = model.loss_fn

	# Create pipeline
	pipe = Pipeline(model, sample)
	optimizer = torch.optim.Adam(pipe.parameters())

	# Warmup iterations
	for _ in range(1):
		_ = pipe(sample, target, loss_fn)
		optimizer.step()

	# Profile iterations
	profiler.start()
	cudart().cudaProfilerStart()

	for i in range(1):
		if rank == 0:
			print(f"Iteration {i}")
		_ = pipe(sample, target, loss_fn, profile=True)
		optimizer.step()

	cudart().cudaProfilerStop()
	profiler.stop()

	pipe.clear()

	if dist.is_initialized():
		dist.destroy_process_group()
