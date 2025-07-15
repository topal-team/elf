import argparse
import os
import sys
import torch
import wandb
import datetime
from torch.distributed.fsdp import fully_shard
import torch.nn as nn

sys.path.append("./")
from elf.zb_utils import replace_linear_with_linear_dw
import elf.pipeline as MyPipe
from models.simple import ChainTransformer, FullTransformer
from elf.utils import Timer
import torch.distributed.pipelining as PiPPy


def fsdp_set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
	for i, layer in enumerate(model.blocks):
		if i >= len(model.blocks) - num_to_forward_prefetch:
			break
		layers_to_prefetch = [model.blocks[i + j] for j in range(1, num_to_forward_prefetch + 1)]
		layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def fsdp_set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
	for i, layer in enumerate(model.blocks):
		if i < num_to_backward_prefetch:
			continue
		layers_to_prefetch = [model.blocks[i - j] for j in range(1, num_to_backward_prefetch + 1)]
		layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def reset_parameters(module):
	for c in module.children():
		if getattr(c, "reset_parameters", False):
			c.reset_parameters()
		else:
			reset_parameters(c)


def make_meta_model(args):
	if args.transformer_type == "full":
		with torch.device("meta"):
			model = FullTransformer(
				input_dim=args.input_dim,
				hidden_dim=args.hidden_dim,
				n_blocks=args.nblocks,
				seq_len=args.seq_len,
				num_heads=args.nheads,
				sdp_backend=args.sdp_backend,
			).to(dtype)
	else:
		with torch.device("meta"):
			model = ChainTransformer(
				hidden_dim=args.hidden_dim,
				n_blocks=args.nblocks,
				seq_len=args.seq_len,
				num_heads=args.nheads,
				sdp_backend=args.sdp_backend,
			).to(dtype)
	return model


def send_meta_to_device(model, device="cuda"):
	model.to_empty(device=device)
	reset_parameters(model)
	return model


def get_inputs_and_targets(model, batch_size, args, device):
	kwargs = {"device": device}
	if args.transformer_type == "chain":
		kwargs["dtype"] = dtype
	sample = model.get_sample(batch_size, **kwargs)  # keep as long for embedding
	target = model.get_target(batch_size, **kwargs)

	return sample, target


def gather_results(time, mem):
	world_size = int(os.getenv("WORLD_SIZE"))
	local_rank = int(os.getenv("LOCAL_RANK"))

	mems = [torch.tensor(0.0, device=local_rank) for _ in range(world_size)] if rank == 0 else None
	torch.distributed.gather(torch.tensor(mem, device=local_rank), mems, 0)
	if mems:
		mems = [m.item() for m in mems]

	return time, mems


def get_placement(schedule_type, world_size):
	match schedule_type:
		case "1f1b" | "zbh1" | "afab" | "zbh2":
			placement = list(range(world_size))
		case "megatron":
			placement = list(range(world_size)) * 2
		case "zbv":
			placement = list(range(world_size)) + list(reversed(range(world_size)))
		case _:
			raise ValueError(f"Unknown schedule type '{args.schedule}'")

	return placement


def get_schedule(schedule_type, stages, nmb, loss_fn):
	"""
	Returns the appropriate schedule object for PiPPy based on the schedule type.
	This function maps ELF schedule names to their PiPPy equivalents.

	:param schedule_type: The type of schedule to use (e.g., "1f1b", "zbh1", "zbh2")
	:param stage: PiPPy pipeline stage
	:param n_stages: Number of pipeline stages
	:param nmb_size: Number of microbatches
	:return: A PiPPy schedule object
	"""
	schedule_type = schedule_type.lower()

	match schedule_type:
		case "afab":
			assert len(stages) == 1
			return PiPPy.ScheduleGPipe(stages[0], nmb, loss_fn=loss_fn)
		case "1f1b":
			assert len(stages) == 1
			return PiPPy.Schedule1F1B(stages[0], nmb, loss_fn=loss_fn)
		case "megatron":
			assert len(stages) == 2
			return PiPPy.ScheduleInterleaved1F1B(stages, nmb, loss_fn=loss_fn)
		case "zbh1":
			assert len(stages) == 1
			return PiPPy.ScheduleInterleavedZeroBubble(stages, nmb, loss_fn=loss_fn)
		case "zbv":
			assert len(stages) == 2
			return PiPPy.ScheduleZBVZeroBubble(stages, nmb, loss_fn=loss_fn)
		case _:
			raise ValueError(f"Unknown schedule type '{schedule_type}' for PiPPy")


def get_parts(model, rank, placement):
	parts = []
	blocks_per_stage = len(model.blocks) // len(placement)
	start, end = 0, 0
	for i, p in enumerate(placement):
		end += blocks_per_stage + (1 if i < (len(model.blocks) % len(placement)) else 0)
		if rank != p:
			start = end
			continue

		if isinstance(model, FullTransformer) and i == 0:
			parts.append(send_meta_to_device(nn.Sequential(model.embed, *model.blocks[start:end])))
		elif isinstance(model, FullTransformer) and i == len(placement) - 1:
			parts.append(send_meta_to_device(nn.Sequential(*model.blocks[start:end], model.head)))
		else:
			parts.append(send_meta_to_device(nn.Sequential(*model.blocks[start:end])))

		start = end

	return parts


def find_stage_global_idx(rank, placement, local_idx):
	cpt = 0
	for i, p in enumerate(placement):
		if p == rank:
			if cpt == local_idx:
				return i

			cpt += 1

	return None


def pippy(args):
	model = make_meta_model(args)

	nmb = world_size * args.mb_per_rank
	batch_size = args.mb_size * nmb
	placement = get_placement(args.schedule, world_size)
	parts = get_parts(model, rank, placement)
	n_stages = len(placement)
	stages = [
		PiPPy.PipelineStage(
			p, find_stage_global_idx(rank, placement, i), n_stages, torch.cuda.current_device()
		)
		for i, p in enumerate(parts)
	]
	schedule = get_schedule(args.schedule, stages, nmb, model.loss_fn)

	##TODO add optimizer
	def get_args_kwargs():
		inputs, targets = get_inputs_and_targets(model, batch_size, args, device)

		input_args = []
		input_kwargs = {}
		if rank == placement[0]:
			input_args.append(inputs)
		if rank == placement[-1]:
			input_kwargs["target"] = targets

		return input_args, input_kwargs

	# Warmup
	for _ in range(5):
		input_args, input_kwargs = get_args_kwargs()
		_ = schedule.step(*input_args, **input_kwargs)

	torch.cuda.reset_peak_memory_stats()
	with Timer() as timer:
		for _ in range(args.niters):
			input_args, input_kwargs = get_args_kwargs()
			_ = schedule.step(*input_args, **input_kwargs)

	return timer.time(), torch.cuda.max_memory_allocated() / 2**30


def elf(args):
	model = make_meta_model(args)

	nmb = world_size * args.mb_per_rank
	batch_size = args.mb_size * nmb

	placement = get_placement(args.schedule, world_size)
	parts = get_parts(model, rank, placement)

	scheduler = args.schedule
	if scheduler == "megatron":
		scheduler = "1f1b"  # Not the same name

	for part in parts:
		if scheduler in ["zbh1", "zbv", "zbh2"]:
			replace_linear_with_linear_dw(part, "cpu")

	sources, dsts = MyPipe.get_sources_targets_sequential(placement)
	pipe = MyPipe.Pipeline(
		parts,
		None,
		partitioner=False,
		schedule=scheduler,
		placement=placement,
		sources=sources,
		targets=dsts,
	)
	optimizer = torch.optim.Adam(pipe.parameters(), lr=1e-2)

	# Warmup
	for _ in range(5):
		inputs, targets = get_inputs_and_targets(model, batch_size, args, device)
		y, loss = pipe(inputs, targets, model.loss_fn, split_size=args.mb_size)
		optimizer.step()
		optimizer.zero_grad()

	torch.cuda.reset_peak_memory_stats()
	with Timer() as timer:
		for _ in range(args.niters):
			inputs, targets = get_inputs_and_targets(model, batch_size, args, device)
			y, loss = pipe(inputs, targets, model.loss_fn, split_size=args.mb_size, profile=args.profile)
			optimizer.step()
			optimizer.zero_grad()

	return timer.time(), torch.cuda.max_memory_allocated() / 2**30


def fsdp(args):
	model = make_meta_model(args)

	for layer in model.blocks:
		fully_shard(layer)
	fully_shard(model)

	if args.explicit_prefetching:
		fsdp_set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
		fsdp_set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

	model = send_meta_to_device(model)

	optim = torch.optim.Adam(model.parameters(), lr=1e-2)

	torch.cuda.reset_peak_memory_stats()

	# Warmup
	for _ in range(5):
		for _ in range(5):
			for __ in range(args.mb_per_rank):
				if args.explicit_prefetching:
					model.unshard()
				sample, target = get_inputs_and_targets(model, args.mb_size, args, device)
				loss = model.loss_fn(model(sample), target)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optim.step()
			optim.zero_grad()

	with Timer() as timer:
		for _ in range(args.niters):
			for __ in range(args.mb_per_rank):
				if args.explicit_prefetching:
					model.unshard()
				sample, target = get_inputs_and_targets(model, args.mb_size, args, device)
				loss = model.loss_fn(model(sample), target)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optim.step()
			optim.zero_grad()

	time_spent = timer.time()
	memory_usage = torch.cuda.max_memory_allocated() / 2**30

	return time_spent, memory_usage


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")

	parser.add_argument(
		"--run",
		nargs="+",
		choices=["elf", "fsdp", "torch"],
		default=["elf"],
		help="Which framework to run",
	)
	parser.add_argument(
		"--explicit-prefetching", action="store_true", default=False, help="specific to FSDP"
	)
	parser.add_argument(
		"--schedule",
		type=str,
		default="zbv",
		help="Pipeline schedule type (supported: afab, 1f1b, megatron, zbh1, zbv)",
	)
	parser.add_argument("--profile", action="store_true", help="Enable ELF profiling")
	parser.add_argument("--run-id", type=str, default="test", help="Run ID")

	# Train parameters
	parser.add_argument("--mb-size", type=int, default=2, required=False, help="micro-batch size")
	parser.add_argument(
		"--mb-per-rank", type=int, default=2, required=False, help="number of micro-batches per rank"
	)
	parser.add_argument("--niters", type=int, default=10, help="Number of iterations")

	# Default values are from LLama 3.8
	parser.add_argument(
		"--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"]
	)
	parser.add_argument(
		"--sdp-backend",
		type=str,
		default=None,
		choices=[None, "FLASH_ATTENTION", "MATH", "EFFICIENT_ATTENTION"],
	)
	parser.add_argument("--transformer-type", type=str, default="chain", choices=["full", "chain"])

	parser.add_argument("--nblocks", type=int, default=32, required=False, help="number of blocks")
	parser.add_argument(
		"--hidden-dim", type=int, default=4096, required=False, help="hidden dimension"
	)
	parser.add_argument("--input-dim", type=int, default=2000, required=False, help="input dimension")
	parser.add_argument("--seq-len", type=int, default=1024, required=False, help="sequence length")
	parser.add_argument("--nheads", type=int, default=32, required=False, help="number of attn heads")
	parser.add_argument("--dropout", type=float, default=0.1, required=False, help="dropout value")
	args = parser.parse_args()

	rank = int(os.environ["RANK"])
	local_rank = int(os.environ["LOCAL_RANK"])
	world_size = int(os.getenv("WORLD_SIZE"))
	device = torch.device(f"cuda:{local_rank}")
	torch.cuda.set_device(device)
	torch.distributed.init_process_group(
		backend="nccl", timeout=datetime.timedelta(seconds=10)
	)  ## , device_id=device)
	torch.manual_seed(0)

	match args.dtype:
		case "float32":
			dtype = torch.float32
		case "float16":
			dtype = torch.float16
		case "bfloat16":
			dtype = torch.bfloat16

	if rank == 0:
		batch_size = args.mb_size * args.mb_per_rank * world_size
		config_dict = {
			"pp_size": world_size,
			"batch_size": batch_size,
			"seq_len": args.seq_len,
			"niters": args.niters,
		}

		wandb.init(
			project="compare-frameworks",
			entity="topal-inria",
			id=f"{args.run_id}_{world_size}",
			group=args.run_id,
			job_type="framework-comparison",
			config=config_dict,
			mode="offline",
		)

	if rank == 0:
		dummy = make_meta_model(args)

		# Log metrics
		metrics = {
			"nb_ranks": world_size,
			"n_blocks": args.nblocks,
			"parameters": sum(p.numel() for p in dummy.parameters()),
		}

		del dummy

	funcs = {"elf": elf, "fsdp": fsdp, "torch": pippy}

	for framework_name in args.run:
		if rank == 0:
			print("Running Framework", framework_name)

		time, mem = funcs[framework_name](args)
		time, mems = gather_results(time, mem)

		if rank == 0:
			metrics[f"{framework_name}_time"] = time
			metrics[f"{framework_name}_max_mem"] = max(mems)
			# Log memory for each GPU
			for i, mem in enumerate(mems):
				metrics[f"{framework_name}_rank_{i}_mem"] = mem
		torch.distributed.barrier()
		torch.cuda.empty_cache()
		torch.cuda.synchronize()

	if rank == 0:
		wandb.log(metrics)
		wandb.finish()

	torch.distributed.destroy_process_group()
