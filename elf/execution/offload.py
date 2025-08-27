import torch
from typing import Any, Dict, Optional, Tuple

__all__ = ["OffloadToCPU"]


class PinnedHostTensorPool:
	"""Pinned-memory arena allocator for 1D CPU tensors used as flat storages.

	- Maintains large pinned-parent chunks per dtype
	- Sub-allocates non-overlapping views (offset/length) from parents
	- Tracks free segments and coalesces on free
	- Supports proactive reservation of big chunks via reserve()
	"""

	def __init__(
		self, device: torch.device | str = "cpu", amount: int = 0, dtype: torch.dtype = torch.float32
	) -> None:
		self.device = torch.device(device)
		# Parents per dtype
		self._parents_by_dtype: Dict[torch.dtype, list[torch.Tensor]] = {}
		# Free segments per parent id: {id(parent): [(offset, length), ...]} sorted by offset
		self._free_segments: Dict[int, list[Tuple[int, int]]] = {}
		# Map view id -> (parent_tensor, offset, length)
		self._view_to_segment: Dict[int, Tuple[torch.Tensor, int, int]] = {}

		if amount > 0:
			self.reserve(amount, dtype)

	@staticmethod
	def _bucket_size(required_elems: int) -> int:
		# Next power of two to reduce reallocations
		b = 1
		while b < required_elems:
			b <<= 1
		return b

	def _first_fit(
		self, dtype: torch.dtype, request_elems: int
	) -> Optional[Tuple[torch.Tensor, int]]:
		"""Find a first-fit free segment among existing parents for given dtype.

		Returns (parent_tensor, offset) or None if no segment fits.
		"""
		for parent in self._parents_by_dtype.get(dtype, []):
			pid = id(parent)
			free_list = self._free_segments.get(pid, [])
			for i, (off, length) in enumerate(free_list):
				if length >= request_elems:
					# Allocate from the start of this segment
					alloc_off = off
					new_off = off + request_elems
					new_len = length - request_elems
					if new_len == 0:
						del free_list[i]
					else:
						free_list[i] = (new_off, new_len)
					return parent, alloc_off
		return None

	def _alloc_parent(self, dtype: torch.dtype, min_elems: int) -> torch.Tensor:
		"""Allocate a new pinned parent chunk for dtype with at least min_elems."""
		parent_elems = self._bucket_size(min_elems)
		parent = torch.empty(parent_elems, dtype=dtype, device=self.device, pin_memory=True)
		self._parents_by_dtype.setdefault(dtype, []).append(parent)
		self._free_segments[id(parent)] = [(0, parent_elems)]
		return parent

	def _alloc_from_parent(self, parent: torch.Tensor, request_elems: int) -> torch.Tensor:
		"""Allocate a view of request_elems from parent's free list (assumes space exists)."""
		pid = id(parent)
		free_list = self._free_segments.get(pid, [])
		# First entry must have enough space
		off, length = free_list[0]
		alloc_off = off
		new_off = off + request_elems
		new_len = length - request_elems
		if new_len == 0:
			del free_list[0]
		else:
			free_list[0] = (new_off, new_len)
		view = torch.as_strided(parent, size=(request_elems,), stride=(1,), storage_offset=alloc_off)
		self._view_to_segment[id(view)] = (parent, alloc_off, request_elems)
		return view

	def _insert_free(self, parent: torch.Tensor, off: int, length: int) -> None:
		"""Insert a freed segment back into parent's free list (with coalescing)."""
		pid = id(parent)
		lst = self._free_segments.setdefault(pid, [])
		inserted = False
		for i, (seg_off, seg_len) in enumerate(lst):
			if off < seg_off:
				lst.insert(i, (off, length))
				inserted = True
				break
		if not inserted:
			lst.append((off, length))
		# Coalesce adjacent segments
		merged: list[Tuple[int, int]] = []
		for seg_off, seg_len in lst:
			if not merged:
				merged.append((seg_off, seg_len))
			else:
				prev_off, prev_len = merged[-1]
				if prev_off + prev_len == seg_off:
					merged[-1] = (prev_off, prev_len + seg_len)
				else:
					merged.append((seg_off, seg_len))
		self._free_segments[pid] = merged

	def allocate(self, required_elems: int, dtype: torch.dtype) -> torch.Tensor:
		"""Allocate a 1D pinned CPU tensor of required_elems (returns a view)."""
		request_elems = max(1, int(required_elems))
		fit = self._first_fit(dtype, request_elems)
		if fit is None:
			parent = self._alloc_parent(dtype, request_elems)
			# allocate from fresh parent
			return self._alloc_from_parent(parent, request_elems)
		parent, alloc_off = fit
		view = torch.as_strided(parent, size=(request_elems,), stride=(1,), storage_offset=alloc_off)
		self._view_to_segment[id(view)] = (parent, alloc_off, request_elems)
		return view

	def free(self, tensor: torch.Tensor) -> None:
		# Only accept pinned CPU 1D tensors
		if tensor.device.type != "cpu" or not tensor.is_pinned() or tensor.dim() != 1:
			return
		vid = id(tensor)
		seg = self._view_to_segment.pop(vid, None)
		if seg is None:
			# Not a view we created; ignore
			return
		parent, off, length = seg
		self._insert_free(parent, off, length)

	def reserve(self, required_bytes: int, dtype: torch.dtype = torch.float32) -> None:
		"""Proactively allocate and add a pinned parent chunk big enough for required_bytes."""
		itemsize = torch.empty((), dtype=dtype).element_size()
		elems_exact = (int(required_bytes) + itemsize - 1) // itemsize
		self._alloc_parent(dtype, elems_exact)


_GLOBAL_PINNED_POOL: Optional[PinnedHostTensorPool] = (
	PinnedHostTensorPool() if torch.cuda.is_available() else None
)


class OffloadToCPU:
	"""Context-manager that off-loads activation tensors to CPU during the forward pass.

	It relies on ``torch.autograd.graph.saved_tensors_hooks``. When the forward pass
	stores an activation needed for the backward pass, the *save* hook is called. We
	immediately copy the tensor to *target_device* (default: CPU) **once per physical
	storage**. If later tensors that *view* the same storage are saved, we only record
	lightweight metadata (shape/stride/offset).

	During the backward pass the *restore* hook reconstructs the original tensor on
	its original device (typically CUDA). If the saved tensor was a view, we
	re-create the view with ``torch.as_strided`` so that storage is still shared.

	Example
	-------
	>>> model = MyLargeModel().cuda()
	>>> with OffloadToCPU():
	...     out = model(inp.cuda()).sum()
	...     out.backward()
	"""

	def __init__(
		self, target_device: torch.device | str = "cpu", pool: Optional["PinnedHostTensorPool"] = None
	) -> None:
		self.target_device = torch.device(target_device)
		# Maps ``storage_ptr`` -> tensor already moved to *target_device*
		self._storage_map: Dict[int, torch.Tensor] = {}
		# Cache of GPU copies during backward to avoid duplicating transfers
		self._gpu_cache: Dict[int, torch.Tensor] = {}
		self._handle = None  # will hold the context object returned by saved_tensors_hooks
		# Streams/events to overlap transfers
		self._offload_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
		self._prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
		self._offload_events: Dict[int, torch.cuda.Event] = {}
		self._prefetch_events: Dict[int, torch.cuda.Event] = {}
		self._orig_device: Dict[int, torch.device] = {}
		# Track how many elements were captured per storage
		self._storage_len: Dict[int, int] = {}
		# Shared pinned-host tensor pool (optional)
		self._pool = pool or _GLOBAL_PINNED_POOL

	# ---------------------------------------------------------------------
	# Hooks
	# ---------------------------------------------------------------------
	def _save_hook(self, tensor: torch.Tensor) -> Dict[str, Any]:
		"""Hook executed when autograd saves *tensor*.

		Returns a lightweight payload that contains either the off-loaded tensor
		(the first time we encounter its storage) or just view metadata.
		"""
		storage_ptr = tensor.untyped_storage().data_ptr()

		# Compute minimal required span for this view to be representable
		size_t = tuple(tensor.size())
		stride_t = tuple(tensor.stride())
		offset = int(tensor.storage_offset())
		max_index = 0
		for dim, st in zip(size_t, stride_t):
			max_index += (int(dim) - 1) * int(st)
		required_elems = int(offset + max_index + 1)

		need_copy = storage_ptr not in self._storage_map or required_elems > self._storage_len.get(
			storage_ptr, 0
		)
		if need_copy:
			cpu_storage = self._allocate_cpu_storage(required_elems, tensor)
			self._copy_to_cpu_async(cpu_storage, tensor)
			self._storage_map[storage_ptr] = cpu_storage
			self._storage_len[storage_ptr] = required_elems
			self._orig_device[storage_ptr] = tensor.device

		payload = {
			"storage_ptr": storage_ptr,
			"size": tuple(tensor.size()),
			"stride": tuple(tensor.stride()),
			"offset": tensor.storage_offset(),
			"dtype": tensor.dtype,
			"device": tensor.device,  # where the tensor lived originally
		}
		# print(f"Size of storage map: {sum(t.nbytes for t in self._storage_map.values()) / 1024**3 :.2f}GB")
		return payload

	def _restore_hook(self, payload: Dict[str, Any]) -> torch.Tensor:
		"""Recreate the original tensor from *payload* during the backward pass."""
		# print(f"  Storage map: {self._storage_map}")
		# 1. Fetch the flat CPU tensor that owns the storage
		base_cpu = self._storage_map[payload["storage_ptr"]]

		# 2. Ensure a device copy exists. If prefetch in-flight, we'll wait below.
		storage_ptr = payload["storage_ptr"]
		if storage_ptr not in self._gpu_cache:
			self._gpu_cache[storage_ptr] = base_cpu.to(payload["device"], non_blocking=False)

		base_dev = self._gpu_cache[storage_ptr]

		# If a prefetch event exists, ensure current stream waits before using
		if torch.cuda.is_available() and storage_ptr in self._prefetch_events:
			torch.cuda.current_stream().wait_event(self._prefetch_events[storage_ptr])

		# 3. Recreate the (potentially strided) view expected by autograd.
		tensor = torch.as_strided(
			base_dev, size=payload["size"], stride=payload["stride"], storage_offset=payload["offset"]
		)

		return tensor

	def _clear_cache(self):
		# Return CPU buffers to pool when possible
		if self._pool is not None:
			for t in self._storage_map.values():
				# Only pooled tensors are pinned CPU; others will be GC'ed
				if t.device.type == "cpu" and t.is_pinned():
					self._pool.free(t)

		self._storage_map.clear()
		self._gpu_cache.clear()
		self._offload_events.clear()
		self._prefetch_events.clear()
		self._orig_device.clear()
		self._storage_len.clear()

	# ------------------------------ helpers ------------------------------
	def _allocate_cpu_storage(self, required_elems: int, like_tensor: torch.Tensor) -> torch.Tensor:
		"""Allocate (preferably from pool) a 1D CPU tensor with required_elems and proper dtype."""
		use_pinned = torch.cuda.is_available() and self.target_device.type == "cpu"
		if use_pinned and self._pool is not None:
			return self._pool.allocate(required_elems, like_tensor.dtype)
		return torch.empty(
			required_elems, dtype=like_tensor.dtype, device=self.target_device, pin_memory=use_pinned
		)

	def _copy_to_cpu_async(self, cpu_storage: torch.Tensor, src: torch.Tensor) -> None:
		"""Copy src (flattened) into cpu_storage, using offload stream if available."""
		# Create flat unit-stride view on src storage
		flat_src = torch.as_strided(
			src.detach(), size=(cpu_storage.numel(),), stride=(1,), storage_offset=0
		)
		if self._offload_stream is not None and src.is_cuda:
			self._offload_stream.wait_stream(torch.cuda.current_stream())
			with torch.cuda.stream(self._offload_stream):
				cpu_storage.copy_(flat_src, non_blocking=True)
		else:
			cpu_storage.copy_(flat_src, non_blocking=False)

		# Record completion event per storage so prefetch can depend on it
		if torch.cuda.is_available():
			ev = torch.cuda.Event()
			(self._offload_events.setdefault(src.untyped_storage().data_ptr(), ev)).record(
				self._offload_stream or torch.cuda.current_stream()
			)

	def prefetch(self) -> None:
		"""Asynchronously start copying all offloaded storages back to their original device.

		Call this after forward and before backward to overlap H2D with other work.
		"""
		if not torch.cuda.is_available() or self._prefetch_stream is None:
			return

		for storage_ptr, base_cpu in list(self._storage_map.items()):
			if storage_ptr in self._gpu_cache:
				continue

			# Make sure that offload is finished
			off_ev = self._offload_events.get(storage_ptr)
			if off_ev is not None:
				self._prefetch_stream.wait_event(off_ev)

			self._prefetch_stream.wait_stream(
				torch.cuda.current_stream()
			)  # wait for current stream to finish (so that the prefetching starts when we want, and not before!)

			with torch.cuda.stream(self._prefetch_stream):
				dev = self._orig_device.get(storage_ptr, torch.device("cuda"))
				gpu_tensor = base_cpu.to(dev, non_blocking=True)
				self._gpu_cache[storage_ptr] = gpu_tensor
				ev = torch.cuda.Event()
				ev.record(self._prefetch_stream)
				self._prefetch_events[storage_ptr] = ev

	# ------------------------------------------------------------------
	# Context-manager plumbing
	# ------------------------------------------------------------------
	def __enter__(self) -> "OffloadToCPU":
		# Register autograd hooks. ``saved_tensors_hooks`` returns a context-manager
		# itself; we keep it so that we can call __exit__ on it later.
		self._handle = torch.autograd.graph.saved_tensors_hooks(self._save_hook, self._restore_hook)
		self._handle.__enter__()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		# Leave the autograd hook context first.
		if self._handle is not None:
			self._handle.__exit__(exc_type, exc_val, exc_tb)
			self._handle = None

		# Propagate exceptions (if any).
		return False

	# Public API to release pooled memory and internal maps once this offloader is no longer needed (typically after the corresponding backward finished).
	def release(self) -> None:
		self._clear_cache()
