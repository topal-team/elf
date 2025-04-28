from contextlib import contextmanager

from torch.utils.checkpoint import checkpoint

from .zb_utils import LayerDW


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
						return checkpoint(original, *args, **kwargs, use_reentrant=True)

					setattr(module, "forward", wrapped_forward)
			try:
				yield
			finally:
				for name, module in self.block.model.named_modules():
					if remat_strategy(name, module):
						# Delete unwanted activations
						for submodule in module.modules():
							if isinstance(submodule, LayerDW):
								submodule.delete("input", mb_id)
						# Restore original forwards
						setattr(module, "forward", getattr(module, "_elf_original_forward"))
						delattr(module, "_elf_original_forward")

		return _selective_remat()

	def register_forward_hooks(self, rbb_strategy, mb_id):
		"""
		Register forward hooks based on the RBB strategy.
		When partially recomputing the gradients for a module, we need the last intermediate activation for which we kept gradients. We call the module that produced this activation the "frontier".
		This function registers a forward hook for the frontier module to save this activation in the block's output variables queue.

		:param rbb_strategy: Strategy function that takes a module name and module, and returns a tuple of two booleans. The first boolean indicates if the module's gradients should be recomputed, and the second boolean indicates if the module is the frontier.
		:type rbb_strategy: Callable[[str, nn.Module], Tuple[bool, bool]]
		:param mb_id: Micro-batch ID
		:type mb_id: int
		:return: List of hook handles
		:rtype: List[torch.utils.hooks.RemovableHandle]
		"""
		handles = []
		if rbb_strategy is not None:
			for name, module in self.block.model.named_modules():
				is_recomputed, is_frontier = rbb_strategy(name, module)
				if is_frontier:

					def forward_hook(
						module, input, output, output_vars=self.block.output_variables, mb_id=mb_id
					):
						output = (output,) if not isinstance(output, tuple) else output
						for out, output_var in zip(output, output_vars):
							output_var[0].set(output_var[0].saved, mb_id, out)

					handle = module.register_forward_hook(forward_hook)
					handles.append(handle)

		return handles

	def register_backward_hooks(self, rbb_strategy, mb_id):
		"""
		Register backward hooks based on the RBB strategy.

		:param rbb_strategy: Strategy function for recomputation during backward
		:param mb_id: Micro-batch ID
		:return: List of hook handles
		"""
		handles = []
		if rbb_strategy is not None:
			for name, module in self.block.model.named_modules():
				is_recomputed, is_frontier = rbb_strategy(name, module)
				if is_frontier:

					def backward_hook(
						module, grad_input, grad_output, output_vars=self.block.output_variables, mb_id=mb_id
					):
						for grad, output_var in zip(grad_output, output_vars):
							output_var[0].set(output_var[0].to_process, mb_id, grad)

					handle = module.register_full_backward_hook(backward_hook)
					handles.append(handle)
		return handles

	def process_after_backward(self, rbf_strategy, rbb_strategy, mb_id):
		"""
		Process modules after backward pass according to strategies.

		:param rbf_strategy: Strategy function for forward recomputation
		:param rbb_strategy: Strategy function for backward recomputation
		:param mb_id: Micro-batch ID
		"""
		for name, module in self.block.model.named_modules():
			if rbf_strategy(name, module):
				for _, submodule in module.named_modules():
					if isinstance(submodule, LayerDW):
						submodule.delete("input", mb_id)

			if rbb_strategy is not None:
				is_recomputed, is_frontier = rbb_strategy(name, module)
				if is_recomputed:
					for _, submodule in module.named_modules():
						if isinstance(submodule, LayerDW):
							submodule.delete("grad_output", mb_id)

	def prepare_recompute_forward(self, rbf_strategy):
		"""
		Prepare modules for forward recomputation.

		:param rbf_strategy: Strategy function for forward recomputation
		"""

		# Deletes the forward function from modules that don't need recomputation (temporarily)
		def delete_forward(name, module, strategy=rbf_strategy):
			# If this module needs recomputation, we need to keep it
			if strategy(name, module):
				return False

			can_be_deleted = True
			for subname, submodule in module.named_children():
				fullname = name + "." + subname if name else subname
				# If any submodule needs recomputation, we need to keep this one too
				if not delete_forward(fullname, submodule, strategy):
					can_be_deleted = False

			if not can_be_deleted:
				return False

			original_forward = getattr(module, "forward")
			setattr(module, "_elf_original_forward", original_forward)
			setattr(module, "forward", lambda *args: args)  # no-op

			return True

		# Remove forward function from modules that don't need recomputation
		delete_forward("", self.block.model, rbf_strategy)

	def restore_forwards(self):
		"""
		Restore original forward functions after recomputation.
		"""
		for _, module in self.block.model.named_modules():
			if hasattr(module, "_elf_original_forward"):
				setattr(module, "forward", getattr(module, "_elf_original_forward"))
				delattr(module, "_elf_original_forward")
