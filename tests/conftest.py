import pytest
import torch


def _has_cuda():
	try:
		return torch.cuda.is_available()
	except Exception:
		return False


def pytest_runtest_setup(item):
	# Skip gpu-tagged tests when CUDA is absent
	if "gpu" in item.keywords and not _has_cuda():
		pytest.skip("CUDA device required for this test")
