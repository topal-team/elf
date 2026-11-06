"""
Glue between ELF and the solvers of the paper (``pipeline-ilps/``).

- :func:`stage_profile_to_ilp` converts a :class:`~elf.profiling.StageProfile` measured by ELF into
  the per-stage format read by ``pipeline-ilps`` (``params.get_params_from_config``).
- :func:`remat_solution` converts a solution written by ``pipeline-ilps/runall.py`` into the
  :class:`~elf.scheduling.ilp.RematSolution` applied by :class:`~elf.scheduling.ilp.RematScheduler`.
- :func:`transformer_units` splits a ``FullTransformer`` into the computation units AdaPipe partitions.
"""

import torch.nn as nn

from elf.profiling import StageProfile
from elf.scheduling.ilp import RematSolution, FORWARD_REMAT_OPTIONS, BACKWARD_REMAT_OPTIONS


def parse_method(method: str) -> tuple[str, str, int]:
	"""Parse ``"<remat>-<scheduler>-<N>W"`` into ``(remat, scheduler, nwaves)``."""
	*remat, scheduler, nwaves = method.split("-")
	return "-".join(remat), scheduler, int(nwaves[:-1])


def stage_profile_to_ilp(profile: StageProfile) -> dict:
	"""
	Per-stage statistics in the format of ``pipeline-ilps``.

	Full forward and full backward rematerialization are added by ``pipeline-ilps`` itself. The only
	profiled option is the recomputation of the activations needed by W: it replays the forward and
	frees the inputs the backward saved for the parameter gradients.
	"""
	held = profile.m_inputs + profile.m_outputs
	return {
		"T": [profile.t_forward, profile.t_backward_inputs, profile.t_backward_params],
		"M": [profile.m_forward, profile.m_backward_inputs, profile.m_backward_params],
		"Mpeak": [
			profile.m_peak_forward,
			profile.m_peak_backward_inputs,
			profile.m_peak_backward_params,
		],
		"Mpeak_no_grad": profile.m_peak_forward_no_grad,
		"Mparams": profile.m_parameters,
		"Tcomm": profile.t_transfer,
		"Minput": profile.m_inputs,
		"Moutput": profile.m_outputs,
		"Mfp": profile.m_forward - held,
		"Mbp": profile.m_backward_inputs - held,
		"forward_remat_options": [],
		"backward_remat_options": [
			{
				"name": "activations_bwd",
				"overhead": profile.t_forward,
				"mem_freed": profile.m_saved_inputs - profile.m_inputs,
			}
		],
	}


def remat_solution(solution: dict, placement: list[int]) -> RematSolution:
	"""
	Rematerialization decisions of a ``pipeline-ilps`` solution, as applied by ELF.

	:param solution: one entry of the ``solutions`` of a file written by ``pipeline-ilps/runall.py``
	:param placement: placement the solution was computed for
	"""

	def binary(matrix):
		return [[int(round(v)) for v in row] for row in matrix]

	return RematSolution(
		placement=list(placement),
		order=[],
		forward_remat={o: binary(solution[o]) for o in FORWARD_REMAT_OPTIONS},
		backward_remat={o: binary(solution[o]) for o in BACKWARD_REMAT_OPTIONS},
		objective=solution["objective"],
		peak_mems=solution["peak_mems"],
	)


class AttentionUnit(nn.Module):
	"""First half of a ``TransformerBlock``: attention and its residual connection."""

	def __init__(self, block):
		super().__init__()
		self.norm1, self.attn, self.proj, self.dropout1 = (
			block.norm1,
			block.attn,
			block.proj,
			block.dropout1,
		)

	def forward(self, x):
		return x + self.dropout1(self.proj(self.attn(self.norm1(x))))


class FeedForwardUnit(nn.Module):
	"""Second half of a ``TransformerBlock``: MLP and its residual connection."""

	def __init__(self, block):
		super().__init__()
		self.norm2, self.mlp, self.dropout2 = block.norm2, block.mlp, block.dropout2

	def forward(self, x):
		return x + self.dropout2(self.mlp(self.norm2(x)))


def transformer_units(model) -> list[tuple[str, nn.Module]]:
	"""
	The units of a ``FullTransformer`` along its forward path: embedding, attention and MLP of every
	block, head. The units share the parameters of ``model`` and compute the same function.
	"""
	units = [("embed", model.embed)]
	for i, block in enumerate(model.blocks):
		units.append((f"block_{i}_attention", AttentionUnit(block)))
		units.append((f"block_{i}_ffn", FeedForwardUnit(block)))
	units.append(("head", model.head))
	return units
