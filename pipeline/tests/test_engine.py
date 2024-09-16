import torch
import torch.nn as nn
from ..pipeline import PipelineBlock
from ..engine import *
import pytest


@pytest.mark.single
def test_compute_loss():
	model = nn.Linear(3, 2)

	inputs = torch.randn((4, 3))
	targets = torch.randn((4, 2))
	loss_fn = nn.functional.cross_entropy

	outputs = model(inputs).detach()
	outputs.requires_grad = True
	loss = loss_fn(outputs, targets, reduction="sum")
	loss.backward()

	block = PipelineBlock(model, 0, ["cpu"], ["inputs"], ["outputs"])
	compute_loss(block, outputs.clone(), targets, loss_fn)
	assert len(block.grads_to_backward) == 1
	assert torch.allclose(block.grads_to_backward[0]["outputs"][1], outputs.grad.data)
