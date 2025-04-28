#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path


def generate_seqlen_configs(base_config_path, min_seqlen, max_seqlen, step):
	"""
	Generate multiple config files with different sequence lengths based on a base config.

	Args:
	    base_config_path: Path to the base config file
	    min_seqlen: Minimum sequence length
	    max_seqlen: Maximum sequence length
	    step: Step size for sequence length increments

	Returns:
	    List of paths to the generated config files
	"""
	# Load the base config
	with open(base_config_path, "r") as f:
		base_config = json.load(f)

	base_name = os.path.basename(base_config_path)
	base_name_without_ext = os.path.splitext(base_name)[0]
	output_dir = os.path.dirname(base_config_path)

	# Create directory for sequence length configs if it doesn't exist
	seqlen_dir = os.path.join(output_dir, "seqlen_configs")
	Path(seqlen_dir).mkdir(exist_ok=True, parents=True)

	generated_configs = []

	# Generate configs for each sequence length
	for seq_len in range(min_seqlen, max_seqlen + 1, step):
		# Create a new config with updated sequence length
		new_config = base_config.copy()
		new_config["model"]["seq_len"] = seq_len

		# Create a filename that includes the sequence length
		new_filename = f"{base_name_without_ext}_seqlen_{seq_len}.json"
		new_path = os.path.join(seqlen_dir, new_filename)

		# Write the new config to file
		with open(new_path, "w") as f:
			json.dump(new_config, f, indent=2)

		generated_configs.append(new_path)
		print(f"Generated config with sequence length {seq_len}: {new_path}")

	return generated_configs


def main():
	parser = argparse.ArgumentParser(description="Generate configs with different sequence lengths")
	parser.add_argument("--base_config", required=True, help="Path to the base config file")
	parser.add_argument("--min_seqlen", type=int, default=128, help="Minimum sequence length")
	parser.add_argument("--max_seqlen", type=int, default=8192, help="Maximum sequence length")
	parser.add_argument(
		"--step", type=int, default=128, help="Step size for sequence length increments"
	)

	args = parser.parse_args()

	generated_configs = generate_seqlen_configs(
		args.base_config, args.min_seqlen, args.max_seqlen, args.step
	)

	print(f"\nGenerated {len(generated_configs)} config files.")


if __name__ == "__main__":
	main()
