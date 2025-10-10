from smk_utils import (
	get_log_dir,
	load_run_params,
	runfile_path,
	list_methods,
	wildcards_to_config_name,
	get_needed_nstages,
	wildcards_to_base_config_name,
	nstages_from_wildcards,
	parse_method,
	find_any_runfile
)

LOG_DIR = get_log_dir(config)


rule strip_runfile:
	"""
	Strip the runfile to only include what's important for profiling, without mention of GPUs.
	"""
	input:
		lambda wildcards: find_any_runfile(wildcards, config)
	output:
		config["RESULTS_DIR"] + "/stripped/{model}-s{sequence_length}-{gpu_type}.json"
	run:
		import json
		with open(input[0], "r") as f:
			data = json.load(f)
		stripped = {k: v for k, v in data.items() if k in ["model", "slurm"]} # keep only what's important for profiling
		with open(output[0], "w") as f:
			json.dump(stripped, f, indent=1)

# Per-method solve: produces a method-scoped solutions file
rule solve_method:
	input:
		profiled=lambda wildcards: config["RESULTS_DIR"] + f"/profiled/{wildcards.model}-s{wildcards.sequence_length}-{wildcards.gpu_type}-{nstages_from_wildcards(wildcards)}stages.json",
		runfile=config["RUNS_DIR"] + "/{model}-s{sequence_length}-{ngpus}{gpu_type}.json"
	output:
		config["RESULTS_DIR"] + "/solutions/{model}-s{sequence_length}-{ngpus}{gpu_type}/{method}.json"
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{ngpus}{gpu_type}.{method}.solve"
	params:
		ngpus=lambda wildcards: load_run_params(config, wildcards).get("ngpus"),
		memgpu=lambda wildcards: load_run_params(config, wildcards).get("memgpu"),
		nmb=lambda wildcards: load_run_params(config, wildcards).get("nmb"),
	shell:
		# Important: use processors+nmb from the original config, not from the profiled one, because we might use profiling of 8 GPUs for wave schedule on 4 GPUs
		"python ../pipeline-ilps/runall.py --config {input.runfile} --profiling {input.profiled} --output {output} --method {wildcards.method} --processors {params.ngpus} --time-limit 32 --mem {params.memgpu} --nmb {params.nmb} 1> {log}.out 2> {log}.err"


# Merge multiple method solutions into a single solutions file
rule merge_solutions:
	input:
		lambda wildcards: expand(
			config["RESULTS_DIR"] + "/solutions/{model}-s{sequence_length}-{ngpus}{gpu_type}/{method}.json",
			model=wildcards.model, sequence_length=wildcards.sequence_length, ngpus=wildcards.ngpus, gpu_type=wildcards.gpu_type,
			method=list_methods(config, wildcards),
			nstages=get_needed_nstages(config, wildcards)
		)
	output:
		config["RESULTS_DIR"] + "/solutions/{model}-s{sequence_length}-{ngpus}{gpu_type}.json"
	wildcard_constraints:
		model=r"[a-zA-Z0-9_-]+",
		sequence_length=r"\d+",
		ngpus=r"\d+",
		gpu_type=r"[a-zA-Z0-9_-]+"
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{ngpus}{gpu_type}.merge_solutions"
	run:
		import json
		merged = {}
		for path in input:
			with open(path, "r") as fh:
				data = json.load(fh)
				for k, v in data.get("solutions", {}).items():
					if "error" in v:
						print(f"Error in {path}: {v['error']}")
						continue
					merged[k] = v
		with open(output[0], "w") as fh:
			json.dump({"solutions": merged}, fh, indent=1)



# ruleorder: merge_benchmarks > merge_benchmarks_partial

# Merge method-level benchmark outputs to the main benchmark file
rule merge_benchmarks:
	input:
		lambda wildcards: expand(
			config["RESULTS_DIR"] + "/benchmarks/{model}-s{sequence_length}-{ngpus}{gpu_type}/{method}.json",
			model=wildcards.model, sequence_length=wildcards.sequence_length, ngpus=wildcards.ngpus, gpu_type=wildcards.gpu_type,
			method=list_methods(config, wildcards),
			nstages=get_needed_nstages(config, wildcards)
		)
	output:
		config["RESULTS_DIR"] + "/benchmarks/{model}-s{sequence_length}-{ngpus}{gpu_type}.json"
	wildcard_constraints:
		model=r"[a-zA-Z0-9_-]+",
		sequence_length=r"\d+",
		ngpus=r"\d+",
		gpu_type=r"[a-zA-Z0-9_-]+"
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{ngpus}{gpu_type}.merge_benchmarks"
	run:
		import json
		merged = {}
		for path in input:
			with open(path, "r") as fh:
				data = json.load(fh)
				if "error" in data:
					print(f"Error in {path}: {data['error']}")
					continue
				merged.update(data)
		with open(output[0], "w") as fh:
			json.dump(merged, fh, indent=1)