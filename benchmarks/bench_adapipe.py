"""
Measure the throughput of AdaPipe (Sun et al., 2024) with a single-wave 1F1B schedule.

AdaPipe (``pipeline-ilps/adapipe.py``) chooses both the partition of the units profiled by
``profile_units.py`` into stages and the units whose activations are recomputed. Recomputed units
are checkpointed in the forward pass. The result is written in the same format as ``bench_remat.py``,
under the key ``AdaPipe-1f1b-1W``.

Usage:
	torchrun --nproc-per-node NGPUS benchmarks/bench_adapipe.py --config-file RUNFILE \
		--profiled-file UNITS --output-file OUTPUT [--mem MB]
"""

import os
import sys
import json
import argparse
import logging
import importlib.util
import statistics

import torch
import torch.distributed as dist

from elf import Placement, replace_layer_with_layer_dw
from elf.registry import SCHEDULERS, resolve
from elf.scheduling import OpOptions, OperationType, RematMode

from benchmark_utils import bench, init_dist, meta_to_device
from models.utils import add_transformer_args, build_model_from_args
from paper_remat import transformer_units

_spec = importlib.util.spec_from_file_location(
	"adapipe",
	os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline-ilps", "adapipe.py"),
)
adapipe = importlib.util.module_from_spec(_spec)
sys.modules["adapipe"] = adapipe
_spec.loader.exec_module(adapipe)

METHOD = "AdaPipe-1f1b-1W"


def parse_args():
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	add_transformer_args(parser)
	parser.add_argument("--profiled-file", type=str, required=True)
	parser.add_argument("--output-file", type=str, required=True)
	parser.add_argument(
		"--mem", type=int, default=None, help="Memory budget in MB (default: memgpu of the runfile)"
	)
	parser.add_argument("--ntrials", type=int, default=None)
	parser.add_argument("--log", choices=["info", "debug", "none"], default="none")
	return parser.parse_args()


def adapipe_args(units, p, nmb, mem):
	return {
		"L": len(units),
		"p": p,
		"n": nmb,
		"M": mem,
		"Mpeak": [u["Mpeak"] for u in units],
		"Mparams": [u["Mparams"] for u in units],
		"tf": [u["T"][0] for u in units],
		"tb": [u["T"][1] for u in units],
		"mem": [u["M"][0] for u in units],
		"minput": [u["Minput"] for u in units],
	}


def adapipe_scheduler(recomputed):
	"""1F1B, where the forward of stage ``s`` checkpoints the modules in ``recomputed[s]``."""
	base = resolve("1f1b", SCHEDULERS)

	def strategy(ids):
		return lambda name, module: RematMode.CHECKPOINT if id(module) in ids else None

	strategies = [strategy(ids) for ids in recomputed]

	def scheduler(placement, nmb, profiles=None):
		schedule = base(placement, nmb, profiles)
		for op in schedule:
			if op.op == OperationType.FORWARD and recomputed[op.block_id]:
				op.options[OpOptions.REMAT_STRATEGY] = strategies[op.block_id]
		return schedule

	return scheduler


def write(path, result):
	with open(path, "w") as f:
		json.dump(result, f, indent=1)


def main():
	args = parse_args()
	if args.log != "none":
		logging.basicConfig(level=getattr(logging, args.log.upper()))
	_, rank, world_size = init_dist()

	with open(args.config_file, "r") as f:
		config = json.load(f)
	with open(args.profiled_file, "r") as f:
		units_stats = json.load(f)["units"]
	nmb = config["nmb"]
	mem = args.mem or config["memgpu"]

	partition = adapipe.solve_adapipe(adapipe_args(units_stats, world_size, nmb, mem))
	if rank == 0:
		for s, i, j, recompute in partition:
			print(f"Stage {s}: units {i} to {j}, recomputed {sorted(recompute)}", flush=True)

	with torch.device("meta"):
		model, dtype = build_model_from_args(args)
		replace_layer_with_layer_dw(model)
	units = [unit for _, unit in transformer_units(model)]

	placement = Placement.default("1f1b", world_size)
	stages = [torch.nn.Sequential(*units[i : j + 1]) for _, i, j, _ in partition]
	recomputed = [{id(units[k]) for k in recompute} for _, _, _, recompute in partition]
	parts = [meta_to_device(stages[s], "cuda") for s in range(len(placement)) if placement[s] == rank]

	try:
		iter_time, peak_mems, all_times = bench(
			model,
			parts,
			adapipe_scheduler(recomputed),
			placement,
			dtype,
			nmb,
			args.ntrials or config.get("ntrials", 1),
		)
	except torch.cuda.OutOfMemoryError:
		print(f"Out of memory on rank {rank} for AdaPipe", flush=True)
		write(args.output_file, {"error": f"Out of memory (rank {rank})"})
		return

	if rank == 0:
		write(
			args.output_file,
			{
				"time": iter_time,
				"peak_mems": [float(m) for m in peak_mems],
				"all_times": all_times,
				"throughput_sps": nmb / statistics.median(all_times),
				"partition": [[s, i, j, sorted(r)] for s, i, j, r in partition],
			},
		)
		print(
			f"{METHOD}: {iter_time:.3f}s / iter, peak memory {[f'{float(m):.1f}' for m in peak_mems]} GB"
		)

	dist.destroy_process_group()


if __name__ == "__main__":
	main()
