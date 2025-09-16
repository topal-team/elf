from smk_utils import (
	get_config_name,
	runfile_path,
	load_run_params,
	get_log_dir,
	build_sbatch_common,
	build_gpu_flag,
	build_sbatch_prefix,
	list_methods,
)

CONFIG_NAME = get_config_name(config)
LOG_DIR = get_log_dir(config)
SBATCH_COMMON = build_sbatch_common(config)
SBATCH_PREFIX = build_sbatch_prefix(config)

rule profile:
	input:
		lambda wildcards: runfile_path(config, wildcards.config_name)
	output:
		config["RESULTS_DIR"] + "/profiling/{config_name}.json"
	resources:
		ngpus=1
	params:
		gpu_flag=lambda wildcards, resources: build_gpu_flag(resources.ngpus, config),
		jobname=lambda wildcards: f"elf-{wildcards.config_name}-profile",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "00:15:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX,
		nstages=lambda wildcards: load_run_params(config, wildcards.config_name).get("ngpus", 4)
	log:
		LOG_DIR + "/{config_name}.profile"
	shell:
		"sbatch --wait {params.sbatch_common} {params.gpu_flag} --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap \"{params.prefix}python profiling.py --config-file {input} --nstages {params.nstages} --output {output} -i 30\""


rule profiling_comms:
	input:
		config["RESULTS_DIR"] + "/profiling/{config_name}.json"
	output:
		config["RESULTS_DIR"] + "/profiled/{config_name}.json"
	resources:
		ngpus=lambda wildcards: load_run_params(config, wildcards.config_name).get("ngpus", 1)
	params:
		gpu_flag=lambda wildcards, resources: build_gpu_flag(4, config),
		jobname=lambda wildcards: f"elf-{wildcards.config_name}-comms",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "00:03:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX
	log:
		LOG_DIR + "/{config_name}.comms"
	shell:
		"sbatch --wait {params.sbatch_common} {params.gpu_flag} --exclusive --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap \"{params.prefix}torchrun --standalone --nproc-per-node=2 profiling-comms.py --config-file {input} --output {output}\""

# Per-method benchmarks in parallel
rule bench_method:
	input:
		sol=lambda wildcards: config["RESULTS_DIR"] + f"/solutions/{wildcards.config_name}/{wildcards.method}.json",
	output:
		config["RESULTS_DIR"] + "/benchmarks/{config_name}/{method}.json"
	resources:
		ngpus=lambda wildcards: load_run_params(config, wildcards.config_name).get("ngpus", 1)
	params:
		# GPUs per node (default 4); can be overridden in SLURM.gpus_per_node
		gpus_per_node=lambda wildcards, resources: int(config.get("SLURM", {}).get("gpus_per_node", 4)),
		# Request per-node GPUs (min of total and per-node)
		gpu_flag=lambda wildcards, resources: build_gpu_flag(resources.ngpus, config),
		# Number of nodes needed: ceil(total_gpus / gpus_per_node)
		nnodes=lambda wildcards, resources: (resources.ngpus + int(config.get("SLURM", {}).get("gpus_per_node", 4)) - 1) // int(config.get("SLURM", {}).get("gpus_per_node", 4)),
		# Single-node nproc-per-node should not exceed total GPUs requested
		nproc_single=lambda wildcards, resources: min(resources.ngpus, int(config.get("SLURM", {}).get("gpus_per_node", 4))),
		# Torchrun flags for single-node vs multi-node runs
		torchrun_flags=lambda wildcards, resources: (
			f"--standalone --nproc-per-node {min(resources.ngpus, int(config.get('SLURM', {}).get('gpus_per_node', 4)))}"
			if ((resources.ngpus + int(config.get('SLURM', {}).get('gpus_per_node', 4)) - 1) // int(config.get('SLURM', {}).get('gpus_per_node', 4))) == 1
			else (
				f"--nnodes {(resources.ngpus + int(config.get('SLURM', {}).get('gpus_per_node', 4)) - 1) // int(config.get('SLURM', {}).get('gpus_per_node', 4))} "
				f"--nproc-per-node {int(config.get('SLURM', {}).get('gpus_per_node', 4))} "
				"--rdzv-id \$SLURM_JOBID --rdzv-backend c10d "
				"--rdzv-endpoint \$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n1)"
			)
		),
		jobname=lambda wildcards: f"elf-{wildcards.config_name}-bench-{wildcards.method}",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "00:45:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX
	log:
		LOG_DIR + "/{config_name}.{method}.bench"
	shell:
		"sbatch --wait {params.sbatch_common} {params.gpu_flag} --exclusive --nodes {params.nnodes} --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap \"{params.prefix} srun torchrun {params.torchrun_flags} -- ../benchmarks/ilps_guided_benchmark.py --restart --solution-file {input.sol} --config-file {input.sol} --output-file {output} --solution-type {wildcards.method}\""

# Merge method-level benchmark outputs to the main benchmark file
rule merge_benchmarks:
	input:
		lambda wildcards: expand(
			config["RESULTS_DIR"] + "/benchmarks/{config_name}/{method}.json",
			config_name=wildcards.config_name,
			method=list_methods(config, wildcards.config_name),
		)
	output:
		config["RESULTS_DIR"] + "/benchmarks/{config_name}.json"
	wildcard_constraints:
		config_name=r"[a-zA-Z0-9_-]+"
	log:
		LOG_DIR + "/{config_name}.merge_benchmarks"
	run:
		import json
		merged = {}
		for path in input:
			with open(path, "r") as fh:
				data = json.load(fh)
				merged.update(data)
		with open(output[0], "w") as fh:
			json.dump(merged, fh, indent=1)