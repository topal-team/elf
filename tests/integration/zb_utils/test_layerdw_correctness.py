import pytest
import torch
import torch.nn as nn

from elf.zb_utils import LinearDW


@pytest.mark.integration
@pytest.mark.gpu
def test_layerdw_correctness():
	"""
	Test case:
	On a model that contains LinearDWs, perform 2 F+B (not W), then W on the first data. Check that model parameters are the same as performing regular F+B on the first data, without LinearDWs.
	"""
	if not torch.cuda.is_available():
		pytest.skip("CUDA not available")

	h = 128
	groundtruth = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)).cuda()

	zb_model = nn.Sequential(LinearDW(nn.Linear(h, h)), nn.ReLU(), LinearDW(nn.Linear(h, h))).cuda()

	def move_last_computed(mb_id):
		for module in zb_model.modules():
			if isinstance(module, LinearDW):
				module.move_last_computed("input", mb_id)
				module.move_last_computed("grad_output", mb_id)

	def bparams(mb_id):
		for module in zb_model.modules():
			if isinstance(module, LinearDW):
				module.backward(mb_id)

	zb_model.load_state_dict(groundtruth.state_dict())

	loss_fn = nn.MSELoss()

	x1 = torch.randn(10, h, device=torch.device("cuda"))
	x2 = torch.randn(10, h, device=torch.device("cuda"))
	z1 = torch.randn(10, h, device=torch.device("cuda"))
	z2 = torch.randn(10, h, device=torch.device("cuda"))

	g1 = groundtruth(x1)
	gt_loss = loss_fn(g1, z1)
	gt_loss.backward()

	# Fwd + Bwd on data 1
	y1 = zb_model(x1)
	assert torch.allclose(y1, g1)
	zb_loss = loss_fn(y1, z1)
	assert torch.allclose(zb_loss, gt_loss)
	zb_loss.backward()
	move_last_computed(0)

	y2 = zb_model(x2)
	zb_loss = loss_fn(y2, z2)
	zb_loss.backward()
	move_last_computed(1)

	bparams(0)

	for p, gp in zip(groundtruth.parameters(), zb_model.parameters()):
		assert torch.allclose(p, gp)

	# Make sure that accumulating gradients works
	bparams(1)

	g2 = groundtruth(x2)
	gt_loss = loss_fn(g2, z2)
	gt_loss.backward()

	for p, gp in zip(groundtruth.parameters(), zb_model.parameters()):
		assert torch.allclose(p, gp)
