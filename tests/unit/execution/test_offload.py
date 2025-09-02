import pytest
import torch

from elf.execution.offload import PinnedHostTensorPool


@pytest.mark.unit
@pytest.mark.gpu
def test_pool_allocate_and_free_coalesce():
	pool = PinnedHostTensorPool(device="cpu", amount=1024, dtype=torch.float32)

	# Allocate two blocks of the same dtype
	t1 = pool.allocate(16, torch.float32)
	t2 = pool.allocate(32, torch.float32)

	# Ensure they are pinned CPU, 1D, and non-overlapping ranges
	assert t1.device.type == "cpu" and t1.is_pinned() and t1.dim() == 1
	assert t2.device.type == "cpu" and t2.is_pinned() and t2.dim() == 1
	assert t1.untyped_storage().data_ptr() == t2.untyped_storage().data_ptr()
	assert (
		t1.storage_offset() + t1.numel() <= t2.storage_offset()
		or t2.storage_offset() + t2.numel() <= t1.storage_offset()
	)

	# Free first, then second, should coalesce into a single large free segment internally.
	pool.free(t1)
	pool.free(t2)

	# Allocate a larger block that fits the coalesced segment
	t_big = pool.allocate(48, torch.float32)
	assert t_big.numel() >= 48
	assert t_big.device.type == "cpu" and t_big.is_pinned()

	# Cleanup
	pool.free(t_big)


@pytest.mark.unit
@pytest.mark.gpu
def test_pool_allocate_without_preallocation():
	"""Test allocator behavior when no memory is pre-reserved."""
	pool = PinnedHostTensorPool(device="cpu")  # No amount specified, no pre-allocation

	# Initial size should be 0 since nothing is pre-allocated
	assert pool.size() == 0

	# First allocation should create a new parent chunk
	t1 = pool.allocate(10, torch.float32)
	assert t1.device.type == "cpu" and t1.is_pinned() and t1.dim() == 1
	assert t1.numel() == 10

	# Pool size should now be non-zero (power of 2 bucket size >= 10 elements)
	size_after_first = pool.size()
	assert size_after_first > 0
	expected_min_bytes = 10 * torch.empty((), dtype=torch.float32).element_size()
	assert size_after_first >= expected_min_bytes

	# Second allocation of same dtype should reuse the same parent if there's space
	t2 = pool.allocate(5, torch.float32)
	assert t2.device.type == "cpu" and t2.is_pinned() and t2.dim() == 1
	assert t2.numel() == 5

	# Pool size should remain the same (no new parent needed)
	assert pool.size() == size_after_first

	# Both tensors should share the same underlying storage
	assert t1.untyped_storage().data_ptr() == t2.untyped_storage().data_ptr()

	# But they should be non-overlapping segments
	assert (
		t1.storage_offset() + t1.numel() <= t2.storage_offset()
		or t2.storage_offset() + t2.numel() <= t1.storage_offset()
	)

	# Large allocation that doesn't fit in remaining space should create new parent
	large_size = size_after_first // torch.empty((), dtype=torch.float32).element_size()
	t3 = pool.allocate(large_size, torch.float32)
	assert t3.device.type == "cpu" and t3.is_pinned() and t3.dim() == 1

	# Pool size should have increased
	size_after_large = pool.size()
	assert size_after_large > size_after_first

	# Clean up
	pool.free(t1)
	pool.free(t2)
	pool.free(t3)


@pytest.mark.gpu
@pytest.mark.unit
def test_pool_reserve_and_size_bucket_rounding():
	pool = PinnedHostTensorPool(device="cpu")

	# Reserve a specific byte size and check size accounts for at least that
	bytes_needed = 1234  # arbitrary
	pool.reserve(bytes_needed, dtype=torch.float32)

	# Size should be at least bytes_needed and a power-of-two bucket due to allocator policy
	size_bytes = pool.size()
	assert size_bytes >= bytes_needed
	# Power of two check
	assert size_bytes & (size_bytes - 1) == 0

	# Allocations should come from the reserved parent without new parent allocation
	# Request a number of elements that fits within the reserved size
	elems = bytes_needed // torch.empty((), dtype=torch.float32).element_size()
	t = pool.allocate(max(1, elems // 2), torch.float32)
	assert t.is_pinned() and t.device.type == "cpu"
	pool.free(t)


@pytest.mark.gpu
@pytest.mark.unit
def test_pool_separate_by_dtype():
	pool = PinnedHostTensorPool(device="cpu")
	t_f32 = pool.allocate(8, torch.float32)
	t_f16 = pool.allocate(8, torch.float16)

	# Different dtypes should lead to different parent arenas
	assert t_f32.untyped_storage().data_ptr() != t_f16.untyped_storage().data_ptr()
	assert t_f32.is_pinned() and t_f16.is_pinned()

	pool.free(t_f32)
	pool.free(t_f16)
