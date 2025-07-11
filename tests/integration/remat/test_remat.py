import pytest
import torch
import torch.nn as nn

from elf.execution.block import PipelineBlock
from elf.partitioners.utils import Signature
from elf.scheduling.scheduling import OpOptions


def _submodule(h):
	return nn.Sequential(
		nn.Linear(h, h, bias=False), nn.ReLU(), nn.Linear(h, h, bias=False), nn.ReLU()
	)


@pytest.mark.integration
@pytest.mark.gpu
def test_selective_remat():
	device = torch.cuda.current_device()
	h = 1024
	b = 1024
	tolerance = (h * b * 4) / 2  # Half of one activation in fp32

	# Create a model with multiple layers
	model = nn.Sequential(_submodule(h), _submodule(h), _submodule(h), _submodule(h)).to(device)

	# Cuda contexts etc
	model(torch.randn(b, h, device=device)).sum().backward()

	placement = [0]
	signature = Signature(inputs=["input"], outputs=["output"], sources=[None], targets=[[0]])
	block = PipelineBlock(
		model=model, id_=0, placement=placement, signature=signature, pp_group=None, dp_group=None
	)
	del model
	x = torch.randn(b, h, device=device, requires_grad=True)

	def remat_strategy(name, module):
		return name in ["1", "2"]

	torch.cuda.reset_peak_memory_stats()
	mem_before_forward_no_remat = torch.cuda.memory_allocated()

	# Forward pass without remat
	input_var = block.input_variables[0]
	input_var.set(input_var.to_process, 0, (None, x))

	block.forward(0)
	memory_after_forward_no_remat = torch.cuda.memory_allocated()

	# Backward pass
	output_var = block.output_variables[0][0]
	output = output_var.get(output_var.saved, 0)
	loss = output.sum()

	loss.backward()
	memory_after_backward_no_remat = torch.cuda.memory_allocated()

	# Clear gradients and reset
	for var in block.input_variables:
		var.clear()
	for var in block.output_variables:
		for dst in var:
			dst.clear()

	del loss
	output.detach_()

	torch.cuda.reset_peak_memory_stats()

	mem_before_forward_with_remat = torch.cuda.memory_allocated() - output.nbytes
	assert abs(mem_before_forward_with_remat - mem_before_forward_no_remat) < tolerance, (
		f"Memory before forward should be the same with and without remat: {mem_before_forward_with_remat / 1024 / 1024:.2f}MB vs {mem_before_forward_no_remat / 1024 / 1024:.2f}MB"
	)

	# Now test with selective remat
	input_var.set(input_var.to_process, 0, (None, x))

	# Forward pass with remat
	options = {OpOptions.REMAT_STRATEGY: remat_strategy}
	block.forward(0, **options)

	memory_after_forward_with_remat = torch.cuda.memory_allocated() - output.nbytes

	# Backward pass
	output2 = output_var.get(output_var.saved, 0)
	loss2 = output2.sum()

	loss2.backward()
	memory_after_backward_with_remat = torch.cuda.memory_allocated() - output.nbytes

	# With remat, forward memory should be lower (activations not stored for recomputed layers)
	assert memory_after_forward_with_remat < memory_after_forward_no_remat, (
		f"Remat should reduce forward memory: {memory_after_forward_with_remat / 1024 / 1024:.2f}MB vs {memory_after_forward_no_remat / 1024 / 1024:.2f}MB"
	)

	assert abs(memory_after_backward_with_remat - memory_after_backward_no_remat) < tolerance, (
		f"Memory after backward should be the same with and without remat: {memory_after_backward_with_remat / 1024 / 1024:.2f}MB vs {memory_after_backward_no_remat / 1024 / 1024:.2f}MB"
	)

	# Verify outputs are the same (correctness check)
	assert torch.allclose(output, output2, atol=1e-5), (
		"Outputs should be identical with and without remat"
	)
