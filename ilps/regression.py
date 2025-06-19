#!/usr/bin/env python
"""
Regression analysis script for transformer performance data.

This script performs linear regression analysis on performance data collected by
the profiling.py script. It extracts timing and memory coefficients for model execution
and updates the configuration file with these values for ILP solvers.

Usage:
    python ilps/regression.py --input-file INPUT_FILE --config-file CONFIG_FILE [--output-file OUTPUT_FILE]

Arguments:
    --input-file: Path to the input JSON file
    --config-file: Path to the configuration file to update (default: ilps/configs/default.json)
    --output-file: Path to save the updated configuration file (defaults to overwriting config-file)
"""

import json
import numpy as np
import argparse
import os
from sklearn.linear_model import LinearRegression

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run regression analysis on stats file")
parser.add_argument("--input-file", type=str, help="Path to the input JSON file")
parser.add_argument(
	"--config-file",
	type=str,
	default="ilps/configs/default.json",
	help="Path to the configuration file to update",
)
parser.add_argument(
	"--output-file",
	type=str,
	help="Path to save the updated configuration file (defaults to overwriting config-file)",
)
parser.add_argument("--nstages", "-n", type=int, required=True, help="Number of stages to create")
args = parser.parse_args()

with open(args.input_file, "r") as f:
	data = json.load(f)

# Load the configuration file
with open(args.config_file, "r") as f:
	config = json.load(f)

# Extract data for regression analysis
no_recompute_data = data["no_recompute"]
recompute_data = data["recompute"]
mfp_data = data["mfp_per_block_batch"]
mbp_data = data["mbp_per_block_batch"]

X_no_recomp = np.array(no_recompute_data["features"])
y_no_recomp = np.array(no_recompute_data["times"])
mem_no_recomp = np.array(no_recompute_data["memory"])
param_size_per_block = np.array(no_recompute_data["param_size_per_block"])

X_recomp = np.array(recompute_data["features"])
y_recomp = np.array(recompute_data["times"])
mem_recomp = np.array(recompute_data["memory"])

# Create interaction feature manually (n_blocks * batch_size)
X_poly_no_recomp = np.array([X_no_recomp[:, 0] * X_no_recomp[:, 1]]).T
X_poly_recomp = np.array([X_recomp[:, 0] * X_recomp[:, 1]]).T

# Get feature names
feature_names = ["n_blocks*batch_size", "bias"]
print(f"Features: {feature_names}")

stage_config = {}
stage_config["Mparams"] = float(np.mean(param_size_per_block))
stage_config["M"] = []

ops = ["f", "b", "w"]
# Update runtime memory metrics in config
# For no recomputation
for i in range(len(ops)):
	stage_config["M"].append(float(np.mean(mem_no_recomp[:, i])))


# For recomputation
sr_mem_kept = float(np.mean(mem_recomp[:, ops.index("f")]))
sr_mem_freed = stage_config["M"][ops.index("f")] - sr_mem_kept


# Run regression for timing metrics and update config
print("\nRegression analysis:")
stage_config["T"] = []
# For no recomputation
for i, name in enumerate(["Tf", "Tb", "Tw"]):
	model = LinearRegression(fit_intercept=False)
	model.fit(X_poly_no_recomp, y_no_recomp[:, i])
	coef = float(model.coef_[0])
	stage_config["T"].append(coef)
	print(f"{name}: {coef}")
	print(f"R-squared: {model.score(X_poly_no_recomp, y_no_recomp[:, i])}")

# For recomputation
model = LinearRegression(fit_intercept=False)
model.fit(X_poly_recomp, y_recomp[:, ops.index("b")])
Tb = float(model.coef_[0])
print(f"Tb: {Tb}")
print(f"R-squared: {model.score(X_poly_recomp, y_recomp[:, ops.index('b')])}")
sr_time_overhead = Tb - stage_config["T"][ops.index("b")]

stage_config["forward_remat_options"] = [
	{"name": "selective_fwd", "overhead": sr_time_overhead, "mem_freed": sr_mem_freed}
]
stage_config["backward_remat_options"] = [
	{
		"name": "activations_bwd",
		"overhead": stage_config["T"][ops.index("f")],
		"mem_freed": float(mfp_data),
	}
]

config = {"model": config["model"], "stages": [stage_config] * args.nstages}

# Save the updated configuration
output_file = args.output_file if args.output_file else args.config_file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
	json.dump(config, f, indent=2)

print(f"\nUpdated configuration saved to {output_file}")
