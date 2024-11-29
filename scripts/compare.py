import os
import torch
import torch.nn as nn
import torch.distributed as dist
import sys

sys.path.append("./")
import pipeline.pipeline as MyPipe
from pipeline.utils import Timer
from models.simple import SimpleTransformer

import torch.distributed.pipelining as PiPPy

placement = [0, 1, 2, 3]
schedule = "1f1b"
model = SimpleTransformer(1024, 1024, 32)
batch_size = 128
inputs = model.get_sample(batch_size)
targets = model.get_target(batch_size)
loss_fn = model.loss_fn
mb_size = batch_size // (max(placement) + 1)

def get_part(rank):
	blocks_per_stage = len(model.blocks) // len(placement)
	if rank == 0:
		return nn.Sequential(model.embed, *model.blocks[:blocks_per_stage]), inputs.clone()[:mb_size]
	elif rank == placement[-1]:
		return nn.Sequential(*model.blocks[-blocks_per_stage:], model.head), torch.randn(mb_size, 64, model.hidden_dim).cuda()
	else:
		return nn.Sequential(*model.blocks[blocks_per_stage * rank:blocks_per_stage * (rank + 1)]), torch.randn(mb_size, 64, model.hidden_dim).cuda()

def pippy():
	part, sample = get_part(rank)
	stage = PiPPy.PipelineStage(part, rank, len(placement), torch.cuda.current_device(), input_args=sample)
	schedule = PiPPy.Schedule1F1B(stage, len(placement), loss_fn=loss_fn)
	# Warmup
	for _ in range(5):
		if rank == 0:
			schedule.step(inputs.clone())
		elif rank == world_size - 1:
			losses = []
			_ = schedule.step(target=targets.clone(), losses=losses)
		else:
			_ = schedule.step()

	with Timer() as timer:
		for _ in range(10):
			if rank == 0:
				schedule.step(inputs.clone())
			elif rank == world_size - 1:
				losses = []
				_ = schedule.step(target=targets.clone(), losses=losses)
			else:
				_ = schedule.step()
	if rank == 0:
		torch.cuda.synchronize()
		print(f"Time taken by torch.distributed.pipelining : {timer.time():.3f}s")


def elf():
	pipe = MyPipe.Pipeline(model, inputs.clone(), placement=placement, schedule=schedule)
	# Warmup
	for _ in range(5):
		y, loss = pipe(inputs.clone(), targets.clone(), loss_fn)

	with Timer() as timer:
		for _ in range(10):
			y, loss = pipe(inputs.clone(), targets.clone(), loss_fn)
	if rank == 0:
		torch.cuda.synchronize()
		print(f"Time taken by our framework : {timer.time():.3f}s")


if __name__ == "__main__":
	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	world_size = int(os.getenv("WORLD_SIZE"))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(backend="nccl")

	model.cuda()
	inputs = inputs.cuda()
	targets = targets.cuda()

	pippy()
	elf()

	if dist.is_initialized():
		dist.destroy_process_group()
