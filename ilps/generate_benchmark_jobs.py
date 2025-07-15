#!/usr/bin/env python3
"""
Helper script to generate benchmark job commands for each n value and solution type.
This script reads the solutions file and generates SLURM job commands for each combination.
"""

import json
import argparse
import os
from typing import List


def generate_job_commands(
	solutions_file: str, config_file: str, output_file: str, ngpus: int = 1, slurm_opts: str = ""
) -> List[str]:
	"""Generate SLURM job commands for each n value and solution type."""
	with open(solutions_file, "r") as f:
		solutions = json.load(f)

	with open(config_file, "r") as f:
		config = json.load(f)
		seqlen = config["model"]["seq_len"]

	commands = []

	for n in solutions:
		for solution_type in solutions[n]:
			# Create a unique job name
			job_name = f"{n}_{solution_type}_{seqlen}"

			# Create the command that will be run by jz.sh
			cmd = (
				f"benchmarks/ilps_guided_benchmark.py "
				f"--config-file {config_file} "
				f"--solution-file {solutions_file} "
				f"--output-file {output_file} "
				f"--n {n} "
				f"--solution-type {solution_type} "
			)

			# Create the sbatch command that uses jz.sh
			sbatch_cmd = (
				f"sbatch {slurm_opts} "
				f"--job-name={job_name} "
				f"--output=logs/{job_name}.out "
				f"--error=logs/{job_name}.err "
				f"--gpus={ngpus} "
				f"--time=00:45:00 "
				f"jz.sh {cmd}"
			)
			commands.append(sbatch_cmd)

	return commands


def main():
	parser = argparse.ArgumentParser(description="Generate benchmark job commands")
	parser.add_argument("--solutions-file", required=True, help="Path to solutions file")
	parser.add_argument("--config-file", required=True, help="Path to config file")
	parser.add_argument("--output-file", required=True, help="Path to output file")
	parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs to use per job")
	parser.add_argument("--slurm-opts", default="", help="Additional SLURM options")
	parser.add_argument("--output-script", help="Path to output shell script")
	args = parser.parse_args()

	# Create logs directory if it doesn't exist
	os.makedirs("logs", exist_ok=True)

	commands = generate_job_commands(
		args.solutions_file, args.config_file, args.output_file, args.ngpus, args.slurm_opts
	)

	if args.output_script:
		# Check if file exists
		if os.path.exists(args.output_script):
			# File exists, just append the commands
			with open(args.output_script, "a") as f:
				for cmd in commands:
					f.write(f"{cmd}\n")
		else:
			# File doesn't exist, create it with header
			with open(args.output_script, "w") as f:
				f.write("#!/bin/bash\n\n")
				f.write(f"if [[ -f {args.output_file} ]]; then\n")
				f.write(f"    echo '!! Warning: {args.output_file} already exists, deleting it.'\n")
				f.write(f"    rm -f {args.output_file}\n")
				f.write("fi\n\n")
				f.write("# Generated benchmark job commands\n\n")
				for cmd in commands:
					f.write(f"{cmd}\n")
			os.chmod(args.output_script, 0o755)
		print(f"Generated job script: {args.output_script}")
	else:
		for cmd in commands:
			print(cmd)


if __name__ == "__main__":
	main()
