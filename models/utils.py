"""
Utility helpers for ELF transformer models.

Several benchmarking / local scripts need to instantiate a
`FullTransformer` or a `ChainTransformer` from command-line flags.  Many
of those scripts currently duplicate the same `argparse` plumbing with
slightly different spellings (`--nblocks` vs `--num-blocks`,
`--n-heads` vs `--num-heads`, …).

This file centralises that logic so that adding a new option or changing
a default only has to be done in one place.

Example
-------
>>> import argparse
>>> from models.utils import add_transformer_args, build_model_from_args
>>> parser = argparse.ArgumentParser()
>>> add_transformer_args(parser, model_type="full")
>>> args = parser.parse_args()
>>> model, dtype = build_model_from_args(args, model_type="full")
"""

from __future__ import annotations

import argparse
import logging
import json

from pathlib import Path
from typing import Literal, Any, Dict

import torch

from models.simple import FullTransformer, ChainTransformer


__all__ = ["add_transformer_args", "build_model_from_args", "model_config_from_args"]

logger = logging.getLogger(__name__)

_ModelType = Literal["full", "chain"]

# Preset configurations for FullTransformer (rough equivalents of Llama model sizes)
_FULL_PRESETS = {
	"8b": {"hidden_dim": 4096, "ffn_dim": 14336, "n_blocks": 32, "num_heads": 32},
	"70b": {"hidden_dim": 8192, "ffn_dim": 28672, "n_blocks": 80, "num_heads": 64},
	"405b": {"hidden_dim": 16384, "ffn_dim": 53248, "n_blocks": 126, "num_heads": 128},
}


def get_dtype(dtype):
	match dtype:
		case "float16" | "fp16":
			return torch.float16
		case "bfloat16" | "bf16":
			if not torch.cuda.is_bf16_supported():
				logging.warning("Bfloat16 is not supported on this GPU")
			return torch.bfloat16
		case "float32" | "fp32":
			return torch.float32
		case _:
			raise ValueError(f"Invalid data type: {dtype}")


def get_sdpa(sdpa):
	match sdpa:
		case "none":
			return None
		case "math":
			return "MATH"
		case "flash":
			return "FLASH_ATTENTION"
		case "efficient":
			return "EFFICIENT_ATTENTION"
		case "cudnn":
			return "CUDNN_ATTENTION"
		case "fatt3":
			return "fatt3"
		case _:
			raise ValueError(f"Invalid SDPA implementation: {sdpa}")


def _load_config_file(path: str) -> Dict[str, Any]:
	"""Load a JSON or YAML configuration file and return it as a dictionary.

	The config file is expected to have the following structure:
	{
		"model": {
			"hidden_dim": ...,
			...
		},
	}

	The file can contain any subset of the model configuration parameters.
	The CLI flags will ultimately take precedence over these values.
	.. warning::
		The keys in the config file must match the parameter names in the model, not the CLI flags.
	"""
	file_path = Path(path)
	if not file_path.exists():
		raise FileNotFoundError(f"Config file '{path}' does not exist")

	ext = file_path.suffix.lower()
	if ext == ".json":
		with file_path.open("r", encoding="utf-8") as fh:
			return json.load(fh)
	elif ext in (".yml", ".yaml"):
		try:
			import yaml  # type: ignore
		except ModuleNotFoundError as exc:
			raise ModuleNotFoundError(
				"PyYAML is required to read YAML configuration files. Install it with `pip install pyyaml`."
			) from exc
		with file_path.open("r", encoding="utf-8") as fh:
			return yaml.safe_load(fh)
	else:
		raise ValueError(f"Unsupported config file extension '{ext}'. Use .json, .yml or .yaml.")


def _add_common_args(parser: argparse.ArgumentParser, include_input_dim: bool) -> None:
	"""Register the arguments that are shared by both transformer variants.

	Passing *include_input_dim* adds the `--input-dim` flag that is only
	meaningful for the *FullTransformer* architecture.
	"""

	add = parser.add_argument

	if include_input_dim:
		add("--vocab-size", type=int, help="Vocabulary size (only relevant for FullTransformer)")

	# Optional preset (only relevant for FullTransformer)
	if include_input_dim:  # presets are only defined for the *full* architecture
		add(
			"--config",
			choices=list(_FULL_PRESETS.keys()),
			help="Preset model configuration (overridden by explicit flags)",
		)

	# External configuration file
	add(
		"--config-file",
		dest="config_file",
		type=str,
		metavar="FILE",
		help="Path to a JSON or YAML configuration file (overridden by explicit flags)",
	)

	# Hidden dimension of the model
	add("--hidden-dim", type=int, help="Hidden dimension of the transformer")

	# Number of blocks / layers
	add(
		"--nblocks",
		"--num-blocks",
		dest="nblocks",
		type=int,
		help="Number of transformer blocks/layers",
	)

	# Sequence length
	add("--seq-len", "--seqlen", type=int, help="Sequence length")

	# Attention heads
	add(
		"--num-heads",
		"--nheads",
		"--n-heads",
		dest="num_heads",
		type=int,
		help="Number of attention heads",
	)

	# Dropout
	add("--dropout", type=float, default=0.1, help="Dropout probability")

	# FFN dimension
	add(
		"--ffn-dim",
		type=int,
		default=None,
		help="Dimension of the feed-forward network (defaults to 4 * hidden_dim)",
	)

	# Weight precision / dtype requested by the user.  Most scripts convert
	# this string with their own helper (e.g. `elf.utils.get_dtype`).
	add(
		"--dtype",
		type=str,
		default=None,
		choices=["float16", "bfloat16", "float32", "fp16", "fp32", "bf16"],
		help="Floating-point precision to instantiate the model parameters with",
	)

	# SDPA backend
	add(
		"--sdp-backend",
		type=str,
		default=None,
		choices=["flash", "math", "efficient", "cudnn", "none", "fatt3"],
		help="Scaled-dot-product attention backend to use (torch 2.1+). Special cases: 'fatt3' for Flash Attention 3 from flash-attn library",
	)


def add_transformer_args(parser: argparse.ArgumentParser, *, model_type: _ModelType) -> None:
	"""Insert the appropriate CLI flags for *model_type* into *parser*.

	Parameters
	----------
	parser: argparse.ArgumentParser
	    The parser you are adding the options to.
	model_type: {"full", "chain"}
	    Which model architecture the script will instantiate.
	"""

	if model_type not in ("full", "chain"):
		raise ValueError(f"Unknown model_type '{model_type}'. Expected 'full' or 'chain'.")

	_add_common_args(parser, include_input_dim=(model_type == "full"))


# -----------------------------------------------------------------------------
# Model construction helpers
# -----------------------------------------------------------------------------


def model_config_from_args(args: argparse.Namespace, *, model_type: _ModelType) -> Dict[str, Any]:
	"""Convert *args* (as returned by *argparse*) to a kwargs mapping.

	This is mostly an internal helper that removes the `argparse` loader
	from the front-line scripts.  Returning a *dict* also allows users
	to further tweak the configuration before the model is built.
	"""

	# Start from preset if requested and available
	cfg: Dict[str, Any] = {}

	if model_type == "full" and getattr(args, "config", None):
		cfg.update(_FULL_PRESETS[args.config])

	# Load external config file if provided
	if getattr(args, "config_file", None):
		try:
			cfg.update(_load_config_file(args.config_file)["model"])
		except KeyError as e:
			raise ValueError(
				f"Error loading config file '{args.config_file}. Make sure it exists and has a 'model' key: {e}"
			)

	# Fill/override with CLI values
	cfg.update(
		{
			"hidden_dim": args.hidden_dim if args.hidden_dim is not None else cfg.get("hidden_dim"),
			"n_blocks": args.nblocks if args.nblocks is not None else cfg.get("n_blocks"),
			"seq_len": args.seq_len if args.seq_len is not None else cfg.get("seq_len"),
			"num_heads": args.num_heads if args.num_heads is not None else cfg.get("num_heads"),
			"dropout": args.dropout if args.dropout is not None else cfg.get("dropout"),
			"ffn_dim": args.ffn_dim if args.ffn_dim is not None else cfg.get("ffn_dim", None),
			"sdp_backend": args.sdp_backend if args.sdp_backend is not None else cfg.get("sdp_backend"),
			"dtype": args.dtype if args.dtype is not None else cfg.get("dtype"),
		}
	)

	# Defaults are added here and not in the argparse section because otherwise
	# if the config file specifies them, they will get overwritten.
	if "sdp_backend" in cfg:
		cfg["sdp_backend"] = get_sdpa(cfg["sdp_backend"] or "none")
	if "dtype" in cfg:
		cfg["dtype"] = get_dtype(cfg["dtype"] or "float32")

	if cfg["ffn_dim"] is None:
		logger.warning(f"ffn_dim is not set, using 4 * hidden_dim = {cfg['hidden_dim'] * 4}")
		cfg["ffn_dim"] = cfg["hidden_dim"] * 4

	if model_type == "full" and args.vocab_size is not None:
		cfg["input_dim"] = args.vocab_size

	if model_type == "chain":
		cfg.pop("input_dim", None)

	return cfg


def build_model_from_args(
	args: argparse.Namespace, *, model_type: _ModelType
) -> tuple[torch.nn.Module, torch.dtype]:
	"""Convenience utility that instantiates and returns the requested model as well as the dtype.

	Example
	-------
	>>> parser = argparse.ArgumentParser()
	>>> add_transformer_args(parser, model_type="chain")
	>>> args = parser.parse_args([])  # use defaults
	>>> model, dtype = build_model_from_args(args, model_type="chain")
	"""

	cfg = model_config_from_args(args, model_type=model_type)
	dtype = cfg.pop("dtype")

	if model_type == "full":
		return FullTransformer(**cfg).to(dtype), dtype
	else:  # chain
		# `input_dim` is absent from cfg in this branch by construction.
		return ChainTransformer(**cfg).to(dtype), dtype
