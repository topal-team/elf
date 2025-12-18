"""
Rematerialization manager
"""

from contextlib import contextmanager
import functools

from torch.utils.checkpoint import (
	checkpoint,
	create_selective_checkpoint_contexts,
	CheckpointPolicy,
)

from ..zb_utils import LayerDW


class RematManager:
	"""
	Manages recomputation strategies for backward passes.
	Encapsulates the logic for handling RBF and RBB strategies.
	"""

	def __init__(self, block):
		self.block = block

	def apply_selective_remat(self, remat_strategy, mb_id):
		"""
		Context manager to apply checkpointing based on a forward recomputation strategy.

		:param remat_strategy: Strategy function that takes a module name and module, and returns a boolean indicating if the module should be recomputed.
		:type remat_strategy: Callable[[str, nn.Module], bool]

		.. warning::
		    The strategy should return true only for the root modules that are recomputed. For example, if you have a Transformer block that contains some Linear, and you want to recompute it, the strategy should return True for the Transformer block, but False for the Linear.

		:param mb_id: Micro-batch ID
		:type mb_id: int
		:return: Context manager
		:rtype: contextlib.ContextManager
		"""

		@contextmanager
		def _selective_remat():
			# Save original forwards and wrap with checkpoint
			for name, module in self.block.model.named_modules():
				if remat_strategy(name, module):
					original = getattr(module, "forward")
					setattr(module, "_elf_original_forward", original)

					# Be careful about the scope of "original" here!
					def wrapped_forward(*args, original=original, **kwargs):
						# use_reentrant=True does not work when the inputs do not require grad (e.g. int inputs)
						# https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/20
						return checkpoint(original, *args, **kwargs, use_reentrant=False)

					setattr(module, "forward", wrapped_forward)
			try:
				yield
			finally:
				for name, module in self.block.model.named_modules():
					if remat_strategy(name, module):
						# Delete unwanted activations
						for submodule in module.modules():
							if isinstance(submodule, LayerDW) and not getattr(
								submodule.ctx["input"][mb_id], "is_model_input", False
							):
								submodule.delete("input", mb_id)
						# Restore original forwards
						setattr(module, "forward", getattr(module, "_elf_original_forward"))
						delattr(module, "_elf_original_forward")

		return _selective_remat()


def recompute_all_context_fn():
	"""
	Create a context that recomputes all activations
	"""

	def policy_fn(*args, **kwargs):
		return CheckpointPolicy.MUST_RECOMPUTE

	return functools.partial(create_selective_checkpoint_contexts, policy_fn)
