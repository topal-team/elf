from smk_utils import (
	load_run_params,
	get_log_dir,
	build_sbatch_common,
	build_gpu_flag,
	build_sbatch_prefix,
	build_torchrun_flags,
	compute_nnodes,
)

LOG_DIR = get_log_dir(config)
SBATCH_COMMON = build_sbatch_common(config)
SBATCH_PREFIX = build_sbatch_prefix(config)


rule profile:
	"""
	Profile the model split into the given number of stages, on 2 GPUs (stages on the first one,
	transfer times between the two). GPU-count agnostic: runfiles with different ngpus reuse the output.
	"""
	input:
		config["RESULTS_DIR"] + "/stripped/{model}-s{sequence_length}-{gpu_type}.json"
	output:
		config["RESULTS_DIR"] + "/profiled/{model}-s{sequence_length}-{gpu_type}-{nstages}stages.json",
	resources:
		ngpus=2
	params:
		gpu_flag=lambda wildcards, resources: build_gpu_flag(resources.ngpus, config),
		jobname=lambda wildcards: f"elf-{wildcards.model}-s{wildcards.sequence_length}-{wildcards.gpu_type}-profile",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "00:30:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{gpu_type}-{nstages}stages.profile"
	shell:
		"sbatch --wait {params.sbatch_common} {params.gpu_flag} --exclusive --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap \"{params.prefix}torchrun --standalone --nproc-per-node 2 ../benchmarks/profile_stages.py --config-file {input} --nstages {wildcards.nstages} --output {output}\""


rule profile_units:
	"""
	Profile the units of the model (embedding, attention, MLP, head) for AdaPipe, on 1 GPU.
	"""
	input:
		config["RESULTS_DIR"] + "/stripped/{model}-s{sequence_length}-{gpu_type}.json"
	output:
		config["RESULTS_DIR"] + "/profiled-units/{model}-s{sequence_length}-{gpu_type}.json"
	resources:
		ngpus=1
	params:
		gpu_flag=lambda wildcards, resources: build_gpu_flag(resources.ngpus, config),
		jobname=lambda wildcards: f"elf-{wildcards.model}-s{wildcards.sequence_length}-{wildcards.gpu_type}-profile-units",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "00:15:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{gpu_type}.profile-units"
	shell:
		"sbatch --wait {params.sbatch_common} {params.gpu_flag} --exclusive --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap \"{params.prefix}torchrun --standalone --nproc-per-node 1 ../benchmarks/profile_units.py --config-file {input} --output {output}\""


rule bench_method:
	input:
		solution=config["RESULTS_DIR"] + "/solutions/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}/{method}.json",
		runfile=config["RUNS_DIR"] + "/{model}-s{sequence_length}-{ngpus}{gpu_type}.json"
	output:
		config["RESULTS_DIR"] + "/benchmarks/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}/{method}.json"
	resources:
		ngpus=lambda wildcards: load_run_params(config, wildcards).get("ngpus", 1)
	params:
		gpu_flag=lambda wildcards, resources: build_gpu_flag(resources.ngpus, config),
		nnodes=lambda wildcards, resources: compute_nnodes(resources.ngpus, config),
		torchrun_flags=lambda wildcards, resources: build_torchrun_flags(resources.ngpus, config),
		jobname=lambda wildcards: f"elf-{wildcards.model}-s{wildcards.sequence_length}-{wildcards.ngpus}{wildcards.gpu_type}-bench-{wildcards.method}",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "00:45:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}.{method}.bench"
	shell:
		"""
		if grep -q '"error"' {input.solution}; then
			echo '{{"error": "solution_error"}}' > {output}
		else
			sbatch --wait {params.sbatch_common} {params.gpu_flag} --exclusive --nodes {params.nnodes} --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap "{params.prefix} srun torchrun {params.torchrun_flags} -- ../benchmarks/bench_remat.py --config-file {input.runfile} --solution-file {input.solution} --method {wildcards.method} --output-file {output}" || echo '{{"error": "benchmark_failed", "exit_code": '"$?"'}}' > {output}
		fi
		exit 0
		"""


rule benchmark_adapipe:
	input:
		prof=config["RESULTS_DIR"] + "/profiled-units/{model}-s{sequence_length}-{gpu_type}.json",
		runfile=config["RUNS_DIR"] + "/{model}-s{sequence_length}-{ngpus}{gpu_type}.json"
	output:
		config["RESULTS_DIR"] + "/benchmarks-adapipe/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}.json"
	resources:
		ngpus=lambda wildcards: load_run_params(config, wildcards).get("ngpus", 1)
	params:
		gpu_flag=lambda wildcards, resources: build_gpu_flag(resources.ngpus, config),
		nnodes=lambda wildcards, resources: compute_nnodes(resources.ngpus, config),
		torchrun_flags=lambda wildcards, resources: build_torchrun_flags(resources.ngpus, config),
		jobname=lambda wildcards: f"elf-{wildcards.model}-s{wildcards.sequence_length}-{wildcards.ngpus}{wildcards.gpu_type}-adapipe-mem{wildcards.mem}",
		time=lambda wildcards: config.get("SLURM", {}).get("time", "01:30:00"),
		sbatch_common=SBATCH_COMMON,
		prefix=SBATCH_PREFIX
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}.benchmark-adapipe"
	shell:
		"""
		sbatch --wait {params.sbatch_common} {params.gpu_flag} --exclusive --nodes {params.nnodes} --job-name={params.jobname} --time {params.time} --output {log}.out --error {log}.err --wrap "{params.prefix} srun torchrun {params.torchrun_flags} -- ../benchmarks/bench_adapipe.py --config-file {input.runfile} --profiled-file {input.prof} --mem {wildcards.mem} --output-file {output}" || echo '{{"error": "benchmark_failed", "exit_code": '"$?"'}}' > {output}
		exit 0
		"""
