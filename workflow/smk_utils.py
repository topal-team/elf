import json
from pathlib import Path
from typing import Dict, Any


def runfile_path(config: Dict[str, Any], wildcards_or_name) -> str:
	"""Return path to the run JSON.

	Accepts wildcards or a base config name.
	"""
	if "run" in config and config["run"]:
		return str(config["run"])
	config_name = wildcards_to_base_config_name(wildcards_or_name)
	return f"{config['RUNS_DIR']}/{config_name}.json"


def load_run_params(config: Dict[str, Any], wildcards_or_name) -> Dict[str, Any]:
	with open(runfile_path(config, wildcards_or_name), "r") as fh:
		return json.load(fh)


def get_log_dir(config: Dict[str, Any]) -> str:
	return str(config.get("LOG_DIR", "logs"))


# ---------------------------------------------------------------------------
# Slurm helpers
# ---------------------------------------------------------------------------


def build_sbatch_common(config: Dict[str, Any]) -> str:
	slurm = config.get("SLURM", {})
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


def compute_nnodes(ngpus: int, config: Dict[str, Any]) -> int:
	gpus_per_node = int(config.get("SLURM", {}).get("gpus_per_node", 4))
	return (ngpus + gpus_per_node - 1) // gpus_per_node


def build_torchrun_flags(ngpus: int, config: Dict[str, Any]) -> str:
	"""Build torchrun CLI flags for single-node or multi-node launches."""
	gpus_per_node = int(config.get("SLURM", {}).get("gpus_per_node", 4))
	nnodes = compute_nnodes(ngpus, config)
	if nnodes == 1:
		return f"--standalone --nproc-per-node {min(ngpus, gpus_per_node)}"
	return (
		f"--nnodes {nnodes} "
		f"--nproc-per-node {gpus_per_node} "
		"--rdzv-id \\$SLURM_JOBID --rdzv-backend c10d "
		"--rdzv-endpoint \\$(scontrol show hostnames \\$SLURM_JOB_NODELIST | head -n1)"
	)


# ---------------------------------------------------------------------------
# Method / config name helpers
# ---------------------------------------------------------------------------


def list_methods(config: Dict[str, Any], wildcards_or_name) -> list[str]:
	"""Return list of method keys from the run file.

	Format: "{scheduler}-{name}-{waves}W" (e.g. "zbh1-F-StageRemat-1W").
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


def parse_method(method: str) -> tuple[str, str, int]:
	"""Parse "{remat}-{scheduler}-{N}W" into (remat, scheduler, nwaves)."""
	*method_parts, scheduler, nwaves = method.split("-")
	nwaves = int(nwaves[:-1])
	return "-".join(method_parts), scheduler, nwaves


def nstages_from_wildcards(wildcards) -> int:
	return parse_method(wildcards.method)[2] * int(wildcards.ngpus)


def wildcards_to_base_config_name(wildcards) -> str:
	"""Build base config name from wildcards (without nstages).

	Used to locate the run params file: {model}-s{seq}-{ngpus}{gpu_type}
	"""
	return (
		f"{wildcards.model}-s{int(wildcards.sequence_length)}"
		f"-{int(wildcards.ngpus)}{wildcards.gpu_type}"
	)


def find_any_runfile(wildcards, config: Dict[str, Any]) -> str:
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
