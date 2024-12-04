import pytest
import torch
from elf.utils import TensorMetadata, dtypes


@pytest.mark.single
def test_metadata():
	# Test creation from class method from_tensor
	received = torch.tensor([dtypes.index(torch.float32), 2, 3, 4])
	metadata = TensorMetadata.from_tensor(received)

	assert metadata.dtype == torch.float32
	assert metadata.shape == [2, 3, 4]

	# Test initialization by constructor
	t = torch.empty((6, 3, 2), dtype=torch.bfloat16)
	metadata = TensorMetadata(t)
	assert metadata.dtype == torch.bfloat16
	assert metadata.shape == [6, 3, 2]

	# Test tensor representation of metadata
	t = torch.empty((3, 1, 4), dtype=torch.float16)
	metadata = TensorMetadata(t)
	tensor_repr = metadata.to_tensor()
	assert tensor_repr[0] == dtypes.index(torch.float16)
	assert tensor_repr[1] == 3
	assert tensor_repr[2] == 1
	assert tensor_repr[3] == 4
	assert tensor_repr[4] == 0

	# Test buffer creation
	t = torch.empty((8, 5, 2), dtype=torch.int64)
	metadata = TensorMetadata(t)
	buffer = metadata.get_buffer(3)
	assert buffer.dtype == torch.int64
	assert buffer.shape == torch.Size([3, 8, 5, 2])
