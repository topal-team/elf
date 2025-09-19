from smk_utils import (
	get_config_name,
	get_log_dir,
	load_run_params,
	runfile_path,
	list_methods,
)

CONFIG_NAME = get_config_name(config)
LOG_DIR = get_log_dir(config)


# Per-method solve: produces a method-scoped solutions file
rule solve_method:
	input:
		config["RESULTS_DIR"] + "/profiled/{config_name}.json"
	output:
		config["RESULTS_DIR"] + "/solutions/{config_name}/{method}.json"
	log:
		LOG_DIR + "/{config_name}.{method}.solve"
	params:
		ngpus=lambda wildcards: load_run_params(config, wildcards.config_name).get("ngpus", 1),
		scheduler=lambda wildcards: load_run_params(config, wildcards.config_name).get("scheduler", ""),
		memgpu=lambda wildcards: load_run_params(config, wildcards.config_name).get("memgpu", ""),
		nmb=lambda wildcards: load_run_params(config, wildcards.config_name).get("nmb", None),
	shell:
		"python ../pipeline-ilps/runall.py --config {input} --output {output} --method {wildcards.method} --time-limit 600 --scheduler {params.scheduler} --mem {params.memgpu} --nmb {params.nmb} 1> {log}.out 2> {log}.err"


# Merge multiple method solutions into a single solutions file
rule merge_solutions:
	input:
		lambda wildcards: expand(
			config["RESULTS_DIR"] + "/solutions/{config_name}/{method}.json",
			config_name=wildcards.config_name,
			method=list_methods(config, wildcards.config_name),
		)
	output:
		config["RESULTS_DIR"] + "/solutions/{config_name}.json"
	wildcard_constraints:
		config_name=r"[a-zA-Z0-9_-]+"
	log:
		LOG_DIR + "/{config_name}.merge_solutions"
	run:
		import json
		merged = {}
		for path in input:
			with open(path, "r") as fh:
				data = json.load(fh)
				for k, v in data.get("solutions", {}).items():
					merged[k] = v
		with open(output[0], "w") as fh:
			json.dump({"solutions": merged}, fh, indent=1)



ruleorder: merge_benchmarks > merge_benchmarks_partial

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
				if "error" in data:
					print(f"Error in {path}: {data['error']}")
					continue
				merged.update(data)
		with open(output[0], "w") as fh:
			json.dump(merged, fh, indent=1)

rule merge_benchmarks_partial:
	input:
		lambda wildcards: existing_benchmark_files(config, wildcards)
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
				if "error" in data:
					print(f"Error in {path}: {data['error']}")
					continue
				merged.update(data)
		with open(output[0], "w") as fh:
			json.dump(merged, fh, indent=1)
