import os
import json
from typing import Dict, Any
from snakemake.io import Wildcards


def get_config_name(config: Dict[str, Any]) -> str | None:
	if "config_name" in config and config["config_name"]:
		return str(config["config_name"])
	if "run" in config and config["run"]:
		run_path = str(config["run"])
		return os.path.splitext(os.path.basename(run_path))[0]
	return None


def runfile_path(config: Dict[str, Any], config_name: str) -> str:
	if "run" in config and config["run"]:
		return str(config["run"])
	return f"{config['RUNS_DIR']}/{config_name}.json"


def load_run_params(config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
	with open(runfile_path(config, config_name), "r") as fh:
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


def list_methods(config: Dict[str, Any], config_name: str) -> list[str]:
	"""Return list of methods to run for a given run config.

	The run file can contain a key "methods" listing method keys that will be
	used as solution_type keys in solutions and benchmarks (e.g., "StageRemat",
	"StageRematF", "Uni-F-Remat").
	"""
	params = load_run_params(config, config_name)
	methods = params.get("methods") or []
	if not isinstance(methods, list):
		return []
	return [str(m) for m in methods]


# Put near the top of the file
def existing_benchmark_files(config: Dict[str, Any], wildcards: Wildcards):
	methods = list_methods(config, wildcards.config_name)
	candidates = [
		config["RESULTS_DIR"] + f"/benchmarks/{wildcards.config_name}/{m}.json" for m in methods
	]
	return [p for p in candidates if os.path.exists(p)]
