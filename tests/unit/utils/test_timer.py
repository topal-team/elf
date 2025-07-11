import pytest
import torch

from elf.utils import Timer


def _busy_wait(n: int = 10_000):
	x = 0
	for _ in range(n):
		x += 1
	return x


@pytest.mark.unit
def test_timer_cpu():
	# Basic timing should be non-negative and nested timers should be consistent
	with Timer("cpu") as outer:
		_busy_wait()
		with Timer("cpu") as inner:
			_busy_wait()
		inner_elapsed = inner.time()
	outer_elapsed = outer.time()

	assert inner_elapsed >= 0
	assert outer_elapsed >= inner_elapsed  # outer includes inner


@pytest.mark.unit
@pytest.mark.gpu
def test_timer_gpu():
	if not torch.cuda.is_available():
		pytest.skip("CUDA not available")

	torch.cuda.synchronize()
	with Timer("gpu") as t:
		# simple CUDA workload
		_ = torch.randn(512, 512, device="cuda") * 2
	torch.cuda.synchronize()

	assert t.time() >= 0
