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
	model = SimpleTransformer(2000, 2048, 64, 1024)
	nmb = ws * 2
	mb_size = 2
	batch_size = nmb * mb_size
	# replace_linear_with_linear_dw(model, "cpu")
	sample = model.get_sample(batch_size).cuda()
	target = model.get_target(batch_size).cuda()
	loss_fn = model.loss_fn

	# Create pipeline
	pipe = Pipeline(model, sample, schedule="1f1b")
	optimizer = torch.optim.Adam(pipe.parameters())

	# Warmup iterations
	for _ in range(3):
		_ = pipe(sample, target, loss_fn, split_size=mb_size)
		optimizer.step()

	torch.cuda.synchronize()

	# Profile iterations
	profiler.start()
	cudart().cudaProfilerStart()

	for i in range(30):
		if rank == 0:
			print(f"Iteration {i}")
		_ = pipe(sample, target, loss_fn, split_size=mb_size, profile=True)
		optimizer.step()

	cudart().cudaProfilerStop()
	profiler.stop()

	pipe.clear()

	if dist.is_initialized():
		dist.destroy_process_group()
