#!/usr/bin/env python
"""
Regression analysis script for transformer performance data.

This script performs linear regression analysis on performance data collected by
the profiling.py script. It extracts timing and memory coefficients for model execution
and updates the configuration file with these values for ILP solvers.

Usage:
    python ilps/regression.py --input_file INPUT_FILE --config_file CONFIG_FILE [--output_file OUTPUT_FILE]

Arguments:
    --input_file: Path to the input JSON file with profiling data
    --config_file: Path to the configuration file to update (default: ilps/configs/default.json)
    --output_file: Path to save the updated configuration file (defaults to overwriting config_file)
"""

import json
import numpy as np
import argparse
import os
from sklearn.linear_model import LinearRegression

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run regression analysis on stats file")
parser.add_argument("--input_file", type=str, help="Path to the input JSON file")
parser.add_argument(
	"--config_file",
	type=str,
	default="ilps/configs/default.json",
	help="Path to the configuration file to update",
)
parser.add_argument(
	"--output_file",
	type=str,
	help="Path to save the updated configuration file (defaults to overwriting config_file)",
)
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
peak_mem_no_recomp = np.array(no_recompute_data["peak_memory"])
param_size_per_block = np.array(no_recompute_data["param_size_per_block"])

X_recomp = np.array(recompute_data["features"])
y_recomp = np.array(recompute_data["times"])
mem_recomp = np.array(recompute_data["memory"])
peak_mem_recomp = np.array(recompute_data["peak_memory"])

# Create interaction feature manually (n_blocks * batch_size)
X_poly_no_recomp = np.array([X_no_recomp[:, 0] * X_no_recomp[:, 1]]).T
X_poly_recomp = np.array([X_recomp[:, 0] * X_recomp[:, 1]]).T

# Get feature names
feature_names = ["n_blocks*batch_size", "bias"]
print(f"Features: {feature_names}")

# Update parameter memory in config
config["Mparams"] = float(np.mean(param_size_per_block))

# Update the activation and gradient memory in config
# These are the same with or without recomputation
config["Mfp"] = float(mfp_data)
config["Mbp"] = float(mbp_data)

# Update runtime memory metrics in config
# For no recomputation
for i, name in enumerate(["Mf", "Mb", "Mw"]):
	config[name] = float(np.mean(mem_no_recomp[:, i]))

# For recomputation
for i, name in enumerate(["Mfsr", "Mbsr", "Mwsr"]):
	config[name] = float(np.mean(mem_recomp[:, i]))

# Update peak memory metrics
config["PeakF"] = float(np.mean(peak_mem_no_recomp))
config["PeakFsr"] = float(np.mean(peak_mem_recomp))

# Run regression for timing metrics and update config
print("\nRegression analysis:")
# For no recomputation
for i, name in enumerate(["Tf", "Tb", "Tw"]):
	model = LinearRegression(fit_intercept=False)
	model.fit(X_poly_no_recomp, y_no_recomp[:, i])
	coef = float(model.coef_[0])
	config[name] = coef
	print(f"{name}: {coef}")
	print(f"R-squared: {model.score(X_poly_no_recomp, y_no_recomp[:, i])}")

# For recomputation
for i, name in enumerate(["Tfsr", "Tbsr", "Twsr"]):
	model = LinearRegression(fit_intercept=False)
	model.fit(X_poly_recomp, y_recomp[:, i])
	coef = float(model.coef_[0])
	config[name] = coef
	print(f"{name}: {coef}")
	print(f"R-squared: {model.score(X_poly_recomp, y_recomp[:, i])}")

# Print the new memory metrics
print("\nMemory metrics:")
print(f"Mfp: {config['Mfp']} MB/block/batch")
print(f"Mbp: {config['Mbp']} MB/block/batch")
print(f"PeakF: {config['PeakF']} MB/block/batch")
print(f"PeakFsr: {config['PeakFsr']} MB/block/batch")

# Save the updated configuration
output_file = args.output_file if args.output_file else args.config_file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
	json.dump(config, f, indent=2)

print(f"\nUpdated configuration saved to {output_file}")
