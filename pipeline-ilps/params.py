import json
from utils import generate_schedule


class RematOption:
	class Type:
		FORWARD = "forward"
		BACKWARD = "backward"

	def __init__(self, name, type, mem_freed, time_overhead):
		self.name = name
		self.type = type
		self.mem_freed = mem_freed
		self.time_overhead = time_overhead

	def get_solution(self):
		raise NotImplementedError("Subclasses must implement this method")


def get_default_placement(scheduler, p):
	match scheduler:
		case "hanayo" | "zbv":
			return [i for i in range(p)] + list(reversed([i for i in range(p)]))
		case "megatron":
			return [i for i in range(p)] * 2
		case _:
			return [i for i in range(p)]


def load_config(config_file="configs/default.json"):
	"""
	Load configuration from a JSON file.

	Args:
	    config_name: Name of the configuration file (without .json extension)

	Returns:
	    Dictionary with configuration parameters
	"""
	with open(config_file, "r") as f:
		return json.load(f)


def get_params_from_config_file(config_file):
	"""
	Get parameters for the ILPs from a configuration file.

	Args:
		config_file: Path to the configuration file

	Returns:
		Dictionary with parameters
	"""
	config = load_config(config_file)
	return get_params_from_config(config)


def get_params_from_config(config):
	"""
	Get parameters for the ILPs.

	Args:
		config: Configuration dictionary

	Returns:
		Dictionary with parameters
	"""
	p = config["ngpus"]
	m = config.get("nmb", p * 2)  # number of micro-batches
	MGpu = config["memgpu"]

	scheduler = config.get("scheduler")
	placement = config.get("placement", get_default_placement(scheduler, p))
	sched = generate_schedule(placement, m, scheduler)

	nstages = len(placement)
	assert len(config["stages"]) == nstages, (
		f"Number of stages in config ({len(config['stages'])}) does not match number of stages in placement ({nstages})"
	)

	stages = []
	ops = ["f", "b", "w"]

	for stage_config in config["stages"]:
		T = {k: v for k, v in zip(ops, stage_config["T"])}
		M = {k: v for k, v in zip(ops, stage_config["M"])}
		Mpeak = {k: v for k, v in zip(ops, stage_config["Mpeak"])} if "Mpeak" in stage_config else M
		Mpeak["f_no_grad"] = stage_config["Mpeak_no_grad"]
		Mfp = stage_config["Mfp"]
		Mbp = stage_config["Mbp"]
		Tcomm = stage_config["Tcomm"]
		Mparams = stage_config["Mparams"]
		Minput = stage_config.get("Minput", 0)
		Moutput = stage_config.get("Moutput", 0)

		fwd_remat_options = stage_config["forward_remat_options"]
		fwd_remat_options = [
			RematOption(option["name"], RematOption.Type.FORWARD, option["mem_freed"], option["overhead"])
			for option in fwd_remat_options
		]
		# This option is always possible
		# We need to keep inputs for recompute, and outputs for backward (technically we don't, but it's not implemented in runtime)
		fwd_remat_options.append(
			RematOption("full_fwd", RematOption.Type.FORWARD, M["f"] - Minput - Moutput, T["f"])
		)

		bwd_remat_options = stage_config["backward_remat_options"]
		bwd_remat_options = [
			RematOption(
				option["name"], RematOption.Type.BACKWARD, option["mem_freed"], option["overhead"]
			)
			for option in bwd_remat_options
		]
		# This option is always possible
		# Need to keep both inputs and outputs for recompute
		bwd_remat_options.append(
			RematOption("full_bwd", RematOption.Type.BACKWARD, M["b"] - Moutput - Minput, T["f"] + T["b"])
		)

		stages.append(
			{
				"T": T,
				"M": M,
				"Mpeak": Mpeak,
				"Tcomm": Tcomm,
				"Mparams": Mparams,
				"Mfp": Mfp,
				"Mbp": Mbp,
				"fwd_remat_options": fwd_remat_options,
				"bwd_remat_options": bwd_remat_options,
			}
		)

	params = {
		"p": p,
		"m": m,
		"nstages": nstages,
		"MGpu": MGpu,
		"sched": sched,
		"placement": placement,
		"stages": stages,
	}

	return params
