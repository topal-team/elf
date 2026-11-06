"""
Measure the throughput of a rematerialization strategy computed by ``pipeline-ilps/runall.py``.

The base schedule and the number of waves are given by the method name
(``<remat>-<scheduler>-<N>W``), the rematerialization decisions by the solution file. They are
applied by :class:`elf.scheduling.ilp.RematScheduler`. The model is split as in
``profile_stages.py``. The result is written as ``{method: {time, peak_mems, all_times, throughput_sps}}``,
or ``{"error": ...}`` if there is no solution or the execution runs out of memory.

Usage:
	torchrun --nproc-per-node NGPUS benchmarks/bench_remat.py --config-file RUNFILE \
		--solution-file SOLUTION --method METHOD --output-file OUTPUT
"""

import json
import argparse
import logging
import statistics

import torch
import torch.distributed as dist

from elf import Placement, replace_layer_with_layer_dw
from elf.registry import SCHEDULERS, resolve
from elf.scheduling.ilp import RematScheduler

from benchmark_utils import bench, get_handcrafted_partition, init_dist
from models.utils import add_transformer_args, build_model_from_args
from paper_remat import parse_method, remat_solution


def parse_args():
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	add_transformer_args(parser)
	parser.add_argument("--solution-file", type=str, required=True)
	parser.add_argument("--method", type=str, required=True)
	parser.add_argument("--output-file", type=str, required=True)
	parser.add_argument("--ntrials", type=int, default=None)
	parser.add_argument("--log", choices=["info", "debug", "none"], default="none")
	return parser.parse_args()


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
	with open(args.solution_file, "r") as f:
		solution = json.load(f)["solutions"][args.method]

	if "error" in solution:
		if rank == 0:
			write(args.output_file, {"error": solution["error"]})
		return

	_, scheduler_name, _ = parse_method(args.method)
	placement = Placement.default(scheduler_name, world_size)
	assert len(placement) == len(solution["full_fwd"]), (
		f"Placement {placement} does not match the {len(solution['full_fwd'])} stages of the solution"
	)
	scheduler = RematScheduler(
		resolve(scheduler_name, SCHEDULERS), remat_solution(solution, placement)
	)

	with torch.device("meta"):
		model, dtype = build_model_from_args(args)
		replace_layer_with_layer_dw(model)
	parts = get_handcrafted_partition(model, rank, placement)

	nmb = config["nmb"]
	try:
		iter_time, peak_mems, all_times = bench(
			model, parts, scheduler, placement, dtype, nmb, args.ntrials or config.get("ntrials", 1)
		)
	except torch.cuda.OutOfMemoryError:
		print(f"Out of memory on rank {rank} for {args.method}", flush=True)
		write(args.output_file, {"error": f"Out of memory (rank {rank})"})
		return

	if rank == 0:
		write(
			args.output_file,
			{
				args.method: {
					"time": iter_time,
					"peak_mems": [float(m) for m in peak_mems],
					"all_times": all_times,
					"throughput_sps": nmb / statistics.median(all_times),
				}
			},
		)
		print(
			f"{args.method}: {iter_time:.3f}s / iter, peak memory {[f'{float(m):.1f}' for m in peak_mems]} GB"
		)

	dist.destroy_process_group()


if __name__ == "__main__":
	main()
