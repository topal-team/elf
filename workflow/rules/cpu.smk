from smk_utils import (
	get_log_dir,
	load_run_params,
	list_methods,
	nstages_from_wildcards,
	find_any_runfile,
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
		stripped = {k: v for k, v in data.items() if k in ["model"]}
		with open(output[0], "w") as f:
			json.dump(stripped, f, indent=1)


rule solve_method:
	input:
		profiled=lambda wildcards: config["RESULTS_DIR"] + f"/profiled/{wildcards.model}-s{wildcards.sequence_length}-{wildcards.gpu_type}-{nstages_from_wildcards(wildcards)}stages.json",
		runfile=config["RUNS_DIR"] + "/{model}-s{sequence_length}-{ngpus}{gpu_type}.json"
	output:
		config["RESULTS_DIR"] + "/solutions/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}/{method}.json"
	log:
		LOG_DIR + "/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}.{method}.solve"
	params:
		ngpus=lambda wildcards: load_run_params(config, wildcards).get("ngpus"),
		nmb=lambda wildcards: load_run_params(config, wildcards).get("nmb"),
		solver=config.get("SOLVER", "CPLEX_PY"),
	shell:
		"python ../pipeline-ilps/runall.py --config {input.runfile} --profiling {input.profiled} --output {output} --method {wildcards.method} --processors {params.ngpus} --time-limit 600 --mem {wildcards.mem} --nmb {params.nmb} --solver {params.solver} 1> {log}.out 2> {log}.err"


rule merge_benchmarks:
	input:
		lambda wildcards: expand(
			config["RESULTS_DIR"] + "/benchmarks/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}/{method}.json",
			model=wildcards.model, sequence_length=wildcards.sequence_length, ngpus=wildcards.ngpus, gpu_type=wildcards.gpu_type,
			method=list_methods(config, wildcards),
			mem=wildcards.mem
		)
	output:
		config["RESULTS_DIR"] + "/benchmarks/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}.json"
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


rule merge_all_with_adapipe:
	"""
	Merge all benchmarks with AdaPipe benchmarks into a single comparison file.
	"""
	input:
		ilps=config["RESULTS_DIR"] + "/benchmarks/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}.json",
		adapipe=config["RESULTS_DIR"] + "/benchmarks-adapipe/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}.json"
	output:
		config["RESULTS_DIR"] + "/benchmarks/{model}-s{sequence_length}-{ngpus}{gpu_type}-mem{mem}.with-adapipe.json"
	run:
		import json
		with open(input.ilps, "r") as fh:
			merged = json.load(fh)
		with open(input.adapipe, "r") as fh:
			adapipe = json.load(fh)
		if "error" in adapipe:
			print(f"Error in {input.adapipe}: {adapipe['error']}")
		else:
			merged["AdaPipe-1f1b-1W"] = adapipe
		with open(output[0], "w") as fh:
			json.dump(merged, fh, indent=1)
