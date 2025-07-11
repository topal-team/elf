import pytest
import torch

from elf.pipeline import Pipeline


@pytest.mark.unit
def test_get_mb_sizes():
	pipe = Pipeline.__new__(Pipeline)  # skip init

	batch = [torch.empty(32, 1)]
	split_size = 8
	mb_sizes = pipe._get_mb_sizes(split_size, batch)
	assert mb_sizes == [8, 8, 8, 8]

	batch = [torch.empty(33, 1)]
	mb_sizes = pipe._get_mb_sizes(split_size, batch)
	assert mb_sizes == [8, 8, 8, 8, 1]

	batch = [torch.empty(32, 1)]
	split_size = [5, 9, 11, 7]
	mb_sizes = pipe._get_mb_sizes(split_size, batch)
	assert mb_sizes == [5, 9, 11, 7]
