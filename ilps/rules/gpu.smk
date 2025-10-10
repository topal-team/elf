from smk_utils import (
	runfile_path,
	load_run_params,
	get_log_dir,
	build_sbatch_common,
	build_gpu_flag,
	build_sbatch_prefix,
	existing_benchmark_files,
	wildcards_to_config_name,
	wildcards_to_base_config_name,
	get_needed_nstages,
)

LOG_DIR = get_log_dir(config)
SBATCH_COMMON = build_sbatch_common(config)
SBATCH_PREFIX = build_sbatch_prefix(config)

rule profile:
	"""
	Profile the model on the given number of stages.
	This is number of gpus-agnostic, several runfiles with different number of GPUs can reuse the output.
	"""
	input:
		config["RESULTS_DIR"] + "/stripped/{model}-s{sequence_length}-{gpu_type}.json" # stripped version
	output:
		config["RESULTS_DIR"] + "/profiled/{model}-s{sequence_length}-{gpu_type}-{nstages}stages.json",
	resources:
		ngpus=2
	params:
		gpu_flag=lambda wildcards, resources: build_gpu_flag(resources.ngpus, config),
		jobname=lambda wildcards: f"elf-{wildcards.model}-s{wildcards.sequence_length}-{wildcards.gpu_type}-profile",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "00:15:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{gpu_type}-{nstages}stages.profile"
	shell:
		"sbatch --wait {params.sbatch_common} {params.gpu_flag} --exclusive --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap \"{params.prefix}python profiling.py --config-file {input} --nstages {wildcards.nstages} --output {output} -i 30 && torchrun --standalone --nproc-per-node=2 profiling-comms.py --config-file {output} --output {output}\""

# Per-method benchmarks in parallel
rule bench_method:
	input:
		config["RESULTS_DIR"] + "/solutions/{model}-s{sequence_length}-{ngpus}{gpu_type}/{method}.json"
	output:
		config["RESULTS_DIR"] + "/benchmarks/{model}-s{sequence_length}-{ngpus}{gpu_type}/{method}.json"
	resources:
		ngpus=lambda wildcards: load_run_params(config, wildcards).get("ngpus", 1)
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
		jobname=lambda wildcards: f"elf-{wildcards.model}-s{wildcards.sequence_length}-{wildcards.ngpus}{wildcards.gpu_type}-bench-{wildcards.method}",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "00:45:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{ngpus}{gpu_type}.{method}.bench"
	shell:
		"""
		# Always produce an output file so downstream merge can proceed, even on failure
		if grep -q 'error' {input}; then
			# If solution contains an error, record it directly as the benchmark output
			echo '{{"error": "solution_error"}}' > {output}
		else
			# Run the benchmark via Slurm; on failure, write an error JSON to the output
			sbatch --wait {params.sbatch_common} {params.gpu_flag} --exclusive --nodes {params.nnodes} --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap "{params.prefix} srun torchrun {params.torchrun_flags} -- ../benchmarks/ilps_guided_benchmark.py --restart --solution-file {input} --config-file {input} --output-file {output} --solution-type {wildcards.method}"
			status=$?
			if [ $status -ne 0 ] || [ ! -s {output} ]; then
				echo '{{"error": "benchmark_failed", "exit_code": '"$status"'}}' > {output}
			fi
		fi
		exit 0
		"""

# Memory comparison benchmark with ELF_MEMORY=1
rule memory_comparison:
	input:
		sol=config["RESULTS_DIR"] + "/solutions/{model}-s{sequence_length}-{ngpus}{gpu_type}/{method}.json",
		prof=config["RESULTS_DIR"] + "/profiled/{model}-s{sequence_length}-{gpu_type}.json"
	output:
		config["RESULTS_DIR"] + "/memory_comparison/{model}-s{sequence_length}-{ngpus}{gpu_type}/{method}.json"
	resources:
		ngpus=lambda wildcards: load_run_params(config, wildcards).get("ngpus", 1)
	params:
		gpu_flag=lambda wildcards, resources: build_gpu_flag(resources.ngpus, config),
		nnodes=lambda wildcards, resources: (resources.ngpus + int(config.get("SLURM", {}).get("gpus_per_node", 4)) - 1) // int(config.get("SLURM", {}).get("gpus_per_node", 4)),
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
		jobname=lambda wildcards: f"elf-{wildcards.model}-s{wildcards.sequence_length}-{wildcards.ngpus}{wildcards.gpu_type}-memcomp-{wildcards.method}",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "01:00:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{ngpus}{gpu_type}.{method}.memcomp"
	shell:
		"""
		# Run memory comparison; on failure, emit an error JSON so downstream steps can continue
		sbatch --wait {params.sbatch_common} {params.gpu_flag} --exclusive --nodes {params.nnodes} --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap "{params.prefix} srun torchrun {params.torchrun_flags} -- memory_comparison_benchmark.py --solution-file {input.sol} --config-file {input.prof} --output-file {output} --solution-type {wildcards.method}"
		status=$?
		if [ $status -ne 0 ] || [ ! -s {output} ]; then
			echo '{{"error": "memcomp_failed", "exit_code": '"$status"'}}' > {output}
		fi
		exit 0
		"""