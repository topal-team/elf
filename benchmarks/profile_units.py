"""
Profile the units of a transformer (embedding, attention and MLP of each block, head) for AdaPipe.

Each unit is profiled on its own with :func:`elf.profiling.profile_stage`, without splitting the
backward (AdaPipe only supports 1F1B). All blocks being identical, the units of the first block are
profiled and reused for the others.

Usage:
	torchrun --standalone --nproc-per-node 1 benchmarks/profile_units.py --config-file RUNFILE --output OUTPUT
"""

import json
import argparse

import torch
import torch.distributed as dist

from elf.profiling import profile_stage
from elf.utils import TensorMetadata

from benchmark_utils import init_dist, meta_to_device
from models.utils import add_transformer_args, build_model_from_args
from paper_remat import transformer_units


def parse_args():
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	add_transformer_args(parser)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument("--iterations", "-i", type=int, default=10)
	return parser.parse_args()


def unit_stats(name, profile):
	return {
		"name": name,
		"T": [profile.t_forward, profile.t_backward_inputs + profile.t_backward_params],
		"M": [profile.m_forward, profile.m_backward_inputs],
		"Mpeak": [profile.m_peak_forward, profile.m_peak_backward_inputs],
		"Mparams": profile.m_parameters,
		"Mpeak_no_grad": profile.m_peak_forward_no_grad,
		"Minput": profile.m_inputs,
		"Moutput": profile.m_outputs,
	}


def main():
	args = parse_args()
	init_dist()

	with torch.device("meta"):
		model, dtype = build_model_from_args(args)
	units = transformer_units(model)
	first_block = {"embed", "block_0_attention", "block_0_ffn", "head"}

	profiled = {}
	metas = (TensorMetadata(model.get_sample(1, dtype=dtype, device="cuda")),)
	for name, unit in units:
		if name not in first_block:
			continue
		unit = meta_to_device(unit, "cuda")
		profile, output = profile_stage(unit, metas, args.iterations)
		print(f"{name}: {profile}", flush=True)
		profiled[name] = unit_stats(name, profile)
		metas = (TensorMetadata(output.to("cuda")),)
		unit.zero_grad(set_to_none=True)
		unit.to_empty(device="meta")
		torch.cuda.empty_cache()

	stats = []
	for name, _ in units:
		kind = name if name in ("embed", "head") else "block_0_" + name.split("_", 2)[2]
		stats.append({**profiled[kind], "name": name})

	with open(args.config_file, "r") as f:
		config = json.load(f)
	with open(args.output, "w") as f:
		json.dump({"model": config["model"], "units": stats}, f, indent=1)

	dist.destroy_process_group()


if __name__ == "__main__":
	main()
