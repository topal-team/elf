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
>>> add_transformer_args(parser)
>>> args = parser.parse_args()
>>> model, dtype = build_model_from_args(args)
"""

from __future__ import annotations

import argparse
import logging
import json

from pathlib import Path
from typing import Any, Dict

import torch

from models.simple import FullTransformer, ChainTransformer

__all__ = ["add_transformer_args", "build_model_from_args", "model_config_from_args"]

_ARCHITECTURES = ["full", "chain"]

logger = logging.getLogger(__name__)


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


def _add_common_args(parser: argparse.ArgumentParser) -> None:
	"""Register the arguments that are shared by both transformer variants."""

	add = parser.add_argument

	add("--architecture", type=str, help="Architecture of the model", choices=_ARCHITECTURES)

	add("--vocab-size", type=int, help="Vocabulary size (only relevant for FullTransformer)")

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
	add(
		"--num-kv-heads",
		"--n-kv-heads",
		dest="num_kv_heads",
		type=int,
		help="Number of key/value heads",
		default=None,
		required=False,
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


def add_transformer_args(parser: argparse.ArgumentParser) -> None:
	"""Insert the appropriate CLI flags for a transformer model into *parser*.

	Parameters
	----------
	parser: argparse.ArgumentParser
	    The parser you are adding the options to.
	"""
	_add_common_args(parser)


# -----------------------------------------------------------------------------
# Model construction helpers
# -----------------------------------------------------------------------------


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
	"""Convert *args* (as returned by *argparse*) to a kwargs mapping.

	This is mostly an internal helper that removes the `argparse` loader
	from the front-line scripts.  Returning a *dict* also allows users
	to further tweak the configuration before the model is built.
	"""

	# Start from preset if requested and available
	cfg: Dict[str, Any] = {}

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
			"architecture": args.architecture
			if args.architecture is not None
			else cfg.get("architecture"),
			"hidden_dim": args.hidden_dim if args.hidden_dim is not None else cfg.get("hidden_dim"),
			"n_blocks": args.nblocks if args.nblocks is not None else cfg.get("n_blocks"),
			"seq_len": args.seq_len if args.seq_len is not None else cfg.get("seq_len"),
			"num_heads": args.num_heads if args.num_heads is not None else cfg.get("num_heads"),
			"dropout": args.dropout if args.dropout is not None else cfg.get("dropout"),
			"ffn_dim": args.ffn_dim if args.ffn_dim is not None else cfg.get("ffn_dim", None),
			"dtype": args.dtype if args.dtype is not None else cfg.get("dtype"),
		}
	)

	# Defaults are added here and not in the argparse section because otherwise
	# if the config file specifies them, they will get overwritten.

	backend = args.sdp_backend or cfg.get("sdp_backend", None)
	if backend is not None:
		cfg["sdp_backend"] = get_sdpa(backend)

	if "dtype" in cfg:
		cfg["dtype"] = get_dtype(cfg["dtype"] or "float32")

	if cfg["ffn_dim"] is None:
		logger.warning(f"ffn_dim is not set, using 4 * hidden_dim = {cfg['hidden_dim'] * 4}")
		cfg["ffn_dim"] = cfg["hidden_dim"] * 4

	if args.vocab_size is not None:
		cfg["input_dim"] = args.vocab_size

	if args.num_kv_heads is not None:
		cfg["num_kv_heads"] = args.num_kv_heads

	return cfg


def build_model_from_args(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.dtype]:
	"""Convenience utility that instantiates and returns the requested model as well as the dtype.

	Example
	-------
	>>> parser = argparse.ArgumentParser()
	>>> add_transformer_args(parser)
	>>> args = parser.parse_args([])  # use defaults
	>>> model, dtype = build_model_from_args(args)
	"""

	cfg = model_config_from_args(args)
	dtype = cfg.pop("dtype")
	arch = cfg.pop("architecture")

	if arch not in _ARCHITECTURES:
		raise ValueError(f"Unknown architecture: {arch}")

	match arch:
		case "full":
			return FullTransformer(**cfg).to(dtype), dtype
		case "chain":
			return ChainTransformer(**cfg).to(dtype), dtype
