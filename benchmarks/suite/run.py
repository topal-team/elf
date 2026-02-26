import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import yaml
from torch.utils.flop_counter import FlopCounterMode

from benchmarks.benchmark_utils import get_handcrafted_partition, init_dist, meta_to_device
from elf import (
	Pipeline,
	PipelineConfig,
	Placement,
	get_sources_targets_sequential,
	replace_linear_with_linear_dw,
)
from models.simple import FullTransformer, SimpleResNet

logger = logging.getLogger("benchmark_suite")

MODEL_REGISTRY = {"FullTransformer": FullTransformer, "SimpleResNet": SimpleResNet}

DTYPE_MAP = {
	"float32": torch.float32,
	"fp32": torch.float32,
	"float16": torch.float16,
	"fp16": torch.float16,
	"bfloat16": torch.bfloat16,
	"bf16": torch.bfloat16,
}

GPU_SPECS = {"h100": {"bf16": 989 * 1e12, "fp16": 989 * 1e12, "fp32": 67 * 1e12}}


def normalize_dtype(dtype):
	"""Normalize dtype to string format for GPU specs lookup."""
	if dtype in [torch.bfloat16, "bfloat16", "bf16"]:
		return "bf16"
	elif dtype in [torch.float16, "float16", "fp16"]:
		return "fp16"
	elif dtype in [torch.float32, "float32", "fp32"]:
		return "fp32"
	else:
		return "bf16"


def compute_model_flops(model, model_config: Dict[str, Any], dtype: torch.dtype, batch_size: int):
	"""
	Compute model FLOPs per sample using meta device.

	Returns:
		flops_per_sample: FLOPs for forward + backward pass per sample
	"""
	with torch.device("meta"):
		sample = model.get_sample(batch_size, dtype=dtype)
		target = model.get_target(batch_size, dtype=dtype)

		flop_counter = FlopCounterMode(display=False)
		with flop_counter:
			output = model(sample)
			loss = model.loss_fn(output, target)
			loss.backward()

	total_flops = flop_counter.get_total_flops()

	if hasattr(model, "seq_len"):
		flops_per_sample = total_flops / (batch_size * model.seq_len)
		return flops_per_sample, model.seq_len
	else:
		flops_per_sample = total_flops / batch_size
		return flops_per_sample, None


def load_config(config_path: str) -> Dict[str, Any]:
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
	return config


def validate_config(config: Dict[str, Any]) -> None:
	required_fields = ["models", "scales", "training", "pipeline", "hardware", "output"]
	for field in required_fields:
		if field not in config:
			raise ValueError(f"Missing required field in config: {field}")

	if not config["models"]:
		raise ValueError("No models specified in config")

	if not config["scales"]:
		raise ValueError("No scales specified in config")

	for model_config in config["models"]:
		if "name" not in model_config or "type" not in model_config:
			raise ValueError("Each model must have 'name' and 'type' fields")
		if model_config["type"] not in MODEL_REGISTRY:
			raise ValueError(
				f"Unknown model type: {model_config['type']}. "
				f"Available types: {list(MODEL_REGISTRY.keys())}"
			)


def create_model(model_config: Dict[str, Any], dtype: torch.dtype):
	model_type = model_config["type"]
	model_params = model_config.get("params", {})

	model_class = MODEL_REGISTRY[model_type]

	with torch.device("meta"):
		model = model_class(**model_params)
		if hasattr(model, "to"):
			model = model.to(dtype)

	return model


def run_single_benchmark(
	model,
	model_config: Dict[str, Any],
	world_size: int,
	config: Dict[str, Any],
	rank: int,
	local_rank: int,
) -> Optional[Dict[str, Any]]:
	training_config = config["training"]
	pipeline_config = config["pipeline"]

	batch_size = training_config["batch_size"]
	mb_size = training_config["mb_size"]
	niters = training_config["niters"]
	warmup_iters = training_config["warmup_iters"]
	dtype = DTYPE_MAP[training_config["dtype"]]

	pp = world_size // pipeline_config["dp"]
	scheduler = pipeline_config["scheduler"]
	placement = Placement.default(scheduler, pp)
	partitioner = pipeline_config.get("partitioner", "constrained")

	is_zb_schedule = scheduler in ["zbh1", "zbh2", "zbv"]

	flops_per_sample = None
	seq_len = None
	if rank == 0:
		num_params = sum(p.numel() for p in model.parameters())
		logger.info(
			f"Benchmarking {model_config['name']} on {world_size} GPUs "
			f"({num_params / 1e9:.2f}B parameters)"
		)
		if is_zb_schedule:
			logger.info("Using Zero Bubble schedule - applying replace_linear_with_linear_dw")

		try:
			flops_per_sample, seq_len = compute_model_flops(model, model_config, dtype, 1)
			logger.info(f"Model FLOPs: {flops_per_sample / 1e12:.3f} TFLOPs/sample")
		except Exception as e:
			logger.warning(f"Could not compute model FLOPs: {e}")

	pipe_config = PipelineConfig(
		partitioner=partitioner,
		scheduler=scheduler,
		placement=placement,
		pp=pp,
		dp=pipeline_config["dp"],
	)

	if partitioner == "handcrafted" or partitioner is False:
		parts = get_handcrafted_partition(model, rank, placement)
		if is_zb_schedule:
			for part in parts:
				replace_linear_with_linear_dw(part, "cuda")
		sources, targets = get_sources_targets_sequential(placement)
		pipe_config.partitioner = False
		pipeline = Pipeline(parts, None, config=pipe_config, sources=sources, targets=targets)
	else:
		if rank == 0:
			model = meta_to_device(model, "cuda")
			if is_zb_schedule:
				replace_linear_with_linear_dw(model, "cuda")
			sample_for_pipeline = model.get_sample(mb_size, dtype=dtype, device="cuda")
		else:
			sample_for_pipeline = model.get_sample(mb_size, dtype=dtype)
		pipeline = Pipeline(model, sample_for_pipeline, config=pipe_config)

	for _ in range(warmup_iters):
		pipeline.zero_grad()
		sample = model.get_sample(batch_size, dtype=dtype, device="cuda")
		target = model.get_target(batch_size, dtype=dtype, device="cuda")
		_, _ = pipeline(sample, target, model.loss_fn, split_size=mb_size)

	torch.cuda.reset_peak_memory_stats()
	torch.cuda.synchronize()
	dist.barrier()

	start_time = time.time()

	for _ in range(niters):
		pipeline.zero_grad()
		sample = model.get_sample(batch_size, dtype=dtype, device="cuda")
		target = model.get_target(batch_size, dtype=dtype, device="cuda")
		_, _ = pipeline(sample, target, model.loss_fn, split_size=mb_size)

	torch.cuda.synchronize()
	dist.barrier()
	elapsed_time = time.time() - start_time

	peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
	all_peak_mems = (
		[torch.tensor(0.0, device=local_rank) for _ in range(world_size)] if rank == 0 else None
	)
	dist.gather(torch.tensor(peak_mem_gb, device=local_rank), all_peak_mems, dst=0)

	pipeline.clear()
	torch.cuda.empty_cache()

	if rank == 0:
		peak_mems_list = [m.item() for m in all_peak_mems]
		avg_iter_time = elapsed_time / niters
		throughput = (batch_size * niters) / elapsed_time

		if hasattr(model, "seq_len"):
			throughput_tokens = (batch_size * niters * model.seq_len) / elapsed_time
			throughput_metric = f"{throughput_tokens:.2f} tokens/s"
		else:
			throughput_metric = f"{throughput:.2f} samples/s"

		mfu = None
		theoretical_throughput = None
		if flops_per_sample is not None:
			gpu_type = config["hardware"].get("device_type", "h100")
			dtype_str = normalize_dtype(dtype)
			gpu_flops = GPU_SPECS.get(gpu_type, GPU_SPECS["h100"])[dtype_str]

			if seq_len is not None:
				theoretical_throughput = (world_size * gpu_flops) / flops_per_sample
				mfu = throughput_tokens / theoretical_throughput
			else:
				theoretical_throughput = (world_size * gpu_flops) / flops_per_sample
				mfu = throughput / theoretical_throughput

		log_msg = f"Results for {model_config['name']} on {world_size} GPUs:\n"
		log_msg += f"  Throughput: {throughput_metric}\n"
		if mfu is not None:
			log_msg += f"  MFU: {100 * mfu:.2f}%\n"
			log_msg += f"  Theoretical max: {theoretical_throughput:.2f} {'tokens/s' if seq_len else 'samples/s'}\n"
		log_msg += f"  Avg iteration time: {avg_iter_time:.4f}s\n"
		log_msg += f"  Max memory: {max(peak_mems_list):.2f}GB\n"
		log_msg += f"  Memory per GPU: {peak_mems_list}"

		logger.info(log_msg)

		result = {
			"model": model_config["name"],
			"model_type": model_config["type"],
			"model_config": model_config.get("params", {}),
			"world_size": world_size,
			"throughput_samples_per_sec": throughput,
			"avg_iteration_time": avg_iter_time,
			"peak_memory_per_gpu": peak_mems_list,
			"max_memory": max(peak_mems_list),
			"num_iterations": niters,
			"batch_size": batch_size,
		}

		if hasattr(model, "seq_len"):
			result["throughput_tokens_per_sec"] = throughput_tokens

		if mfu is not None:
			result["mfu"] = mfu
			result["theoretical_throughput"] = theoretical_throughput
			result["model_flops_per_sample"] = flops_per_sample

		return result

	return None


def save_results(
	results: List[Dict[str, Any]], config: Dict[str, Any], rank: int, world_size: int
) -> None:
	if rank != 0:
		return

	output_config = config["output"]
	results_dir = Path(output_config["results_dir"])
	results_dir.mkdir(exist_ok=True)

	base_filename = output_config["results_filename"]
	name_parts = base_filename.rsplit(".", 1)
	if len(name_parts) == 2:
		results_file = results_dir / f"{name_parts[0]}_{world_size}gpu.{name_parts[1]}"
	else:
		results_file = results_dir / f"{base_filename}_{world_size}gpu"

	output_data = {"timestamp": datetime.now().isoformat(), "config": config, "results": results}

	with open(results_file, "w") as f:
		json.dump(output_data, f, indent=2)

	logger.info(f"Results saved to {results_file}")


def log_to_wandb(result: Dict[str, Any], config: Dict[str, Any], run_id: str) -> None:
	try:
		import wandb
	except ImportError:
		logger.warning("wandb not installed, skipping WandB logging")
		return

	wandb_config = config.get("wandb", {})
	if not wandb_config.get("enabled", False):
		return

	run_name = f"{result['model']}_w{result['world_size']}"
	tags = wandb_config.get("tags", []) + [result["model"], f"world_size_{result['world_size']}"]

	wandb.init(
		project=wandb_config.get("project", "elf-benchmark-suite"),
		entity=wandb_config.get("entity"),
		name=run_name,
		group=run_id,
		tags=tags,
		config={
			"model": result["model"],
			"model_type": result["model_type"],
			"model_config": result["model_config"],
			"world_size": result["world_size"],
			"batch_size": result["batch_size"],
			"training_config": config["training"],
			"pipeline_config": config["pipeline"],
		},
	)

	wandb.log(
		{
			"throughput_samples_per_sec": result["throughput_samples_per_sec"],
			"avg_iteration_time": result["avg_iteration_time"],
			"max_memory": result["max_memory"],
			"throughput_tokens_per_sec": result.get("throughput_tokens_per_sec"),
		}
	)

	wandb.finish()


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run benchmark suite for distributed training library"
	)

	parser.add_argument(
		"--config",
		type=str,
		default="benchmarks/suite/config.yaml",
		help="Path to benchmark config YAML file",
	)
	parser.add_argument(
		"--no-wandb", action="store_true", help="Disable WandB logging even if enabled in config"
	)
	parser.add_argument("--models", type=str, nargs="+", help="Filter to specific models (by name)")
	parser.add_argument("--scales", type=int, nargs="+", help="Filter to specific GPU counts")
	parser.add_argument(
		"--log-level",
		type=str,
		default="info",
		choices=["debug", "info", "warning", "error"],
		help="Logging level",
	)

	return parser.parse_args()


def setup_logging(log_level: str, rank: int):
	level_map = {
		"debug": logging.DEBUG,
		"info": logging.INFO,
		"warning": logging.WARNING,
		"error": logging.ERROR,
	}
	level = level_map.get(log_level.lower(), logging.INFO)

	if rank == 0:
		logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
	else:
		logging.basicConfig(level=logging.WARNING)


def main():
	args = parse_args()

	local_rank, rank, world_size = init_dist("nccl")

	setup_logging(args.log_level, rank)

	config = load_config(args.config)
	validate_config(config)

	if args.no_wandb and "wandb" in config:
		config["wandb"]["enabled"] = False

	model_filter = set(args.models) if args.models else None
	scale_filter = set(args.scales) if args.scales else None

	models_to_run = [m for m in config["models"] if model_filter is None or m["name"] in model_filter]
	scales_to_run = [s for s in config["scales"] if scale_filter is None or s in scale_filter]

	if world_size not in scales_to_run:
		if rank == 0:
			logger.warning(
				f"Current world size ({world_size}) is not in the list of scales to benchmark: {scales_to_run}"
			)
			logger.warning("No benchmarks will be run. Please launch with appropriate --nproc_per_node")
		dist.destroy_process_group()
		return

	if rank == 0:
		logger.info(f"Starting benchmark suite with {len(models_to_run)} models")
		logger.info(f"Models: {[m['name'] for m in models_to_run]}")
		logger.info(f"Current scale: {world_size} GPUs")

	run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	all_results = []

	for model_config in models_to_run:
		if rank == 0:
			logger.info(f"\n{'=' * 60}")
			logger.info(f"Benchmarking {model_config['name']}")
			logger.info(f"{'=' * 60}")

		try:
			dtype = DTYPE_MAP[config["training"]["dtype"]]
			model = create_model(model_config, dtype)

			result = run_single_benchmark(model, model_config, world_size, config, rank, local_rank)

			if result is not None:
				all_results.append(result)

				if config.get("wandb", {}).get("enabled", False) and not args.no_wandb:
					log_to_wandb(result, config, run_id)

			del model
			torch.cuda.empty_cache()
			dist.barrier()

		except Exception as e:
			logger.error(f"Error benchmarking {model_config['name']}: {e}")
			import traceback

			traceback.print_exc()
			dist.destroy_process_group()
			raise e

	save_results(all_results, config, rank, world_size)

	if rank == 0:
		logger.info("\n" + "=" * 60)
		logger.info("Benchmark suite completed!")
		logger.info("=" * 60)

	dist.destroy_process_group()


if __name__ == "__main__":
	main()
