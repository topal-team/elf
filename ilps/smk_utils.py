import os
import json
from pathlib import Path
from typing import Dict, Any
from snakemake.io import Wildcards


def _to_base_config_name(arg: Wildcards | str) -> str:
	"""Return base config name (without nstages) from wildcards or string.

	If a string is provided, it's assumed to already be a base config name.
	"""
	if isinstance(arg, Wildcards):
		return wildcards_to_base_config_name(arg)
	return str(arg)


def runfile_path(config: Dict[str, Any], wildcards_or_name: Wildcards | str) -> str:
	"""Return path to the run JSON.

	Accepts wildcards or a base config name.
	"""
	if "run" in config and config["run"]:
		return str(config["run"])
	config_name = _to_base_config_name(wildcards_or_name)
	return f"{config['RUNS_DIR']}/{config_name}.json"


def load_run_params(config: Dict[str, Any], wildcards_or_name: Wildcards | str) -> Dict[str, Any]:
	with open(runfile_path(config, wildcards_or_name), "r") as fh:
		return json.load(fh)


def get_log_dir(config: Dict[str, Any]) -> str:
	return str(config.get("LOG_DIR", "logs"))


def build_sbatch_common(config: Dict[str, Any]) -> str:
	slurm = config.get("SLURM", {})
	# Only include keys that are present; add more mappings as needed per cluster
	arg_map = {
		"partition": "partition",
		"account": "account",
		"qos": "qos",
		"time": "time",
		"cpus_per_task": "cpus-per-task",
		"mem": "mem",
		"constraint": "constraint",
		"nodes": "nodes",
		"reservation": "reservation",
	}
	parts = []
	for key, opt in arg_map.items():
		val = slurm.get(key)
		if val:
			parts.append(f"--{opt}={val}")
	extra = slurm.get("extra")
	if extra:
		parts.append(str(extra))
	return " ".join(parts)


def build_gpu_flag(ngpus: int, config: Dict[str, Any]) -> str:
	slurm = config.get("SLURM", {})
	flag_tpl = slurm.get("gpus_flag", "--gpus={ngpus} --exclusive")
	return flag_tpl.replace("{ngpus}", str(ngpus))


def build_sbatch_prefix(config: Dict[str, Any]) -> str:
	slurm = config.get("SLURM", {})
	setup = str(slurm.get("setup", "")).strip()
	activate = str(slurm.get("venv_activate", "")).strip()

	parts = []
	if setup:
		parts.append(setup)
	if activate:
		parts.append(activate)
	return " ; ".join(parts) + (" ; " if parts else "")


def list_methods(config: Dict[str, Any], wildcards_or_name: Wildcards | str) -> list[str]:
	"""Return list of methods to run for a given run config.

	The run file can contain a key "methods" listing method keys that will be
	used as solution_type keys in solutions and benchmarks (e.g., "StageRemat",
	"StageRematF", "Uni-F-Remat").
	"""
	params = load_run_params(config, wildcards_or_name)
	methods = params.get("methods") or {}
	list_of_methods = []
	for method in methods:
		list_of_methods.extend(
			[
				str(scheduler) + "-" + str(method["name"]) + "-" + str(method["waves"]) + "W"
				for scheduler in method["methods"]
			]
		)

	return list_of_methods


def get_needed_nstages(config: Dict[str, Any], wildcards_or_name: Wildcards | str):
	needed_nstages = set()
	params = load_run_params(config, wildcards_or_name)
	ngpus = params.get("ngpus")
	for method in params.get("methods") or {}:
		waves = method.get("waves")
		needed_nstages.add(waves * ngpus)
	return needed_nstages


def parse_method(method: str) -> tuple[str, str, int]:
	"""Parse the method into its parts.

	Returns: (method, scheduler, nwaves)
	"""
	*method, scheduler, nwaves = method.split("-")
	nwaves = int(nwaves[:-1])
	return "-".join(method), scheduler, nwaves


def nstages_from_wildcards(wildcards: Wildcards) -> int:
	return parse_method(wildcards.method)[2] * int(wildcards.ngpus)


def compose_config_name(
	model: str, sequence_length: str | int, ngpus: str | int, gpu_type: str, nstages: str | int = None
) -> str:
	"""Compose the canonical config name from parts.

	Expected format: {model}-s{sequence_length}-{ngpus}{gpu_type}-{nstages}stages
	"""
	base = f"{str(model)}-s{int(sequence_length)}-{int(ngpus)}{str(gpu_type)}"
	if nstages:
		base += f"-{int(nstages)}stages"
	return base


def wildcards_to_config_name(wildcards: Wildcards) -> str:
	"""Build full config name from wildcards, including nstages when present."""
	fields = ("model", "sequence_length", "ngpus", "gpu_type", "nstages")
	args = [getattr(wildcards, k) for k in fields if hasattr(wildcards, k)]
	return compose_config_name(*args)


def wildcards_to_base_config_name(wildcards: Wildcards) -> str:
	"""Build base config name from wildcards (without nstages).

	Used to locate the run params file, which is named without stages.
	"""
	fields = ("model", "sequence_length", "ngpus", "gpu_type")
	args = [getattr(wildcards, k) for k in fields if hasattr(wildcards, k)]
	return compose_config_name(*args)


# Put near the top of the file
def existing_benchmark_files(config: Dict[str, Any], wildcards: Wildcards):
	cfg_name = wildcards_to_config_name(wildcards)
	methods = list_methods(config, cfg_name)
	candidates = [config["RESULTS_DIR"] + f"/benchmarks/{cfg_name}/{m}.json" for m in methods]
	return [p for p in candidates if os.path.exists(p)]


def find_any_runfile(wildcards: Wildcards, config: Dict[str, Any]) -> str:
	"""Find a runfile with any number of GPUs for a given wildcards."""
	try:
		return next(
			Path(config["RUNS_DIR"]).glob(
				f"{wildcards.model}-s{wildcards.sequence_length}-[0-9]*{wildcards.gpu_type}.json"
			)
		)
	except StopIteration:
		raise FileNotFoundError(
			f"No runfile found for {wildcards.model}-s{wildcards.sequence_length}-{wildcards.gpu_type}.json"
		)
