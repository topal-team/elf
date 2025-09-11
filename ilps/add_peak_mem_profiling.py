import json
import argparse

import torch
from torch.utils.checkpoint import checkpoint

from models.simple import ChainTransformer, Attention, FastAttention
from models.utils import add_transformer_args, model_config_from_args
from elf.zb_utils import replace_linear_with_linear_dw, LayerDW


def parse_args():
	parser = argparse.ArgumentParser()
	add_transformer_args(parser, model_type="chain")
	return parser.parse_args()


def create_model(config, n):
	config["n_blocks"] = n
	dtype = config.pop("dtype")
	with torch.device("cuda"):
		model = ChainTransformer(**config).to(dtype)
	replace_linear_with_linear_dw(model, "cuda")
	config["dtype"] = dtype
	return model, dtype


def bparams(model):
	for module in model.modules():
		if isinstance(module, LayerDW):
			module.move_last_computed("input", 0)
			module.move_last_computed("grad_output", 0)
			module.backward(0)


def zbclear(model):
	for module in model.modules():
		if isinstance(module, LayerDW):
			module.clear()


def apply_checkpointing(model):
	"""Apply selective recomputation via checkpointing to attention layers"""
	for name, module in model.named_modules():
		if isinstance(module, (Attention, FastAttention)):
			original_forward = getattr(module, "forward")

			def wrapped_forward(*args, **kwargs):
				return checkpoint(original_forward, *args, **kwargs, use_reentrant=True)

			setattr(module, "forward", wrapped_forward)
	return model


def main():
	args = parse_args()

	config = model_config_from_args(args, model_type="chain")
	with open(args.config_file, "r") as f:
		data = json.load(f)
		stages = data["stages"]

	n = config["n_blocks"] // len(stages)

	model, dtype = create_model(config, n)

	scenarios = ["f", "b", "w", "fb", "bw", "fbw"]
	peaks = [{scenario: 0 for scenario in scenarios} for _ in range(2)]

	# Warmup
	model(model.get_sample(1, dtype, device="cuda")).sum().backward()
	bparams(model)

	for i in range(2):
		for scenario in scenarios:
			model.zero_grad(set_to_none=False)
			zbclear(model)

			sample = model.get_sample(1, dtype, device="cuda")
			torch.cuda.reset_peak_memory_stats()
			start_mem = torch.cuda.memory_allocated()

			y = model(sample)
			if "f" not in scenario:
				torch.cuda.reset_peak_memory_stats()
				start_mem = torch.cuda.memory_allocated()
				# ignore F

			if "b" or "w" in scenario:
				y.sum().backward()
			if "b" not in scenario and "f" not in scenario:
				torch.cuda.reset_peak_memory_stats()
				start_mem = torch.cuda.memory_allocated()
				# ignore B

			if "w" in scenario:
				bparams(model)

			del y
			torch.cuda.synchronize()
			peak_mem = torch.cuda.max_memory_allocated() - start_mem
			peaks[i][scenario] = peak_mem / 1024 / 1024
			print(
				f"{scenario}: {peaks[i][scenario]:.2f}MB, stored mem: {(torch.cuda.memory_allocated() - start_mem) / (2**20):.2f}MB"
			)
			torch.cuda.empty_cache()

		if i == 0:
			apply_checkpointing(model)
			print("\n-- With checkpointing --\n")

	for stage in stages:
		stage["Mpeaks"] = {"no_recompute": peaks[0], "selective_fwd": peaks[1]}

	with open(args.config_file, "w") as f:
		json.dump(data, f, indent=2)


if __name__ == "__main__":
	main()
