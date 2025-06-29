import torch
import torch.distributed as dist
import os
import sys

sys.path.append("./")
from elf.pipeline import Pipeline
from elf.zb_utils import replace_linear_with_linear_dw
from elf.scheduling import OperationType
from models.simple import FullTransformer

# Initialize distributed training
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group(backend="nccl")

# Create model and pipeline
model = FullTransformer(500, 512, 8, 512, 32, 0.1)
replace_linear_with_linear_dw(model, rank)

batch_size = 32
n_micro_batches = 8
split_size = batch_size // n_micro_batches

inputs = model.get_sample(batch_size)
targets = model.get_target(batch_size)

pipe = Pipeline(model, inputs, list(range(world_size)), scheduler="zbh1")

# Training iteration
model.zero_grad()
y = pipe(inputs, targets, model.loss_fn, split_size=split_size)


detailed_stats = pipe.detailed_stats
# Get memory stats for each operation
memories = detailed_stats["memories"]

# Find all backward_inputs operations and their memory usage
last_mem = 0
for op, mem in memories.items():
	mem = mem / (1024**3)
	if op.op is OperationType.BACKWARD_INPUTS:
		assert mem < last_mem, (
			f"Rank {rank} - Memory increased after {op}: {last_mem:.3f} GB -> {mem:.3f} GB"
		)
		print(
			f"Rank {rank} - Memory check passed for {op}: {last_mem:.3f} GB -> {mem:.3f} GB (- {mem - last_mem:.3f} GB)"
		)
	last_mem = mem

# Clean up
pipe.clear()

dist.barrier()
dist.destroy_process_group()
