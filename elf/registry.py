"""Generic plugin registries used across the **elf** code-base.

The goal is to replace ad-hoc *string ➜ function* dispatch patterns with a
structured, type-safe alternative.  End-users can extend the library by simply
importing :py:mod:`elf.registry` and calling :py:func:`Registry.register`.

Example
-------
>>> from elf.registry import SCHEDULERS
>>> def custom_scheduler(placement, n_micro_batches, signatures):
...     ...
...     return scheduler

>>> SCHEDULERS.register("my_scheduler", custom_scheduler, "My custom scheduler")

Later, inside :py:class:`elf.pipeline.Pipeline` you can refer to it with

>>> pipeline = Pipeline(model, sample, scheduler="my_scheduler")
"""

from __future__ import annotations
from typing import Callable, Dict, Generic, Iterable, List, TypeVar, Protocol, TypeAlias

import torch

import elf

__all__ = ["Registry", "SCHEDULERS", "PARTITIONERS"]

T = TypeVar("T", bound=Callable[..., object])


class Registry(Generic[T]):
	"""Minimal dictionary-backed registry.

	:param name: Human-readable name used in error messages (e.g. *"scheduler"*).
	:type name: str
	"""

	def __init__(self, name: str):
		self._name: str = name
		self._values: Dict[str, T] = {}
		self._descriptions: Dict[str, str] = {}

	# Private helper
	def _add(self, keys: str | Iterable[str], obj: T, description: str = "") -> None:
		"""Internal primitive that inserts *(key → obj)*, with duplicate-check."""
		if isinstance(keys, str):
			keys = [keys]
		for k in keys:
			existing = self._values.get(k)
			if existing is not None and existing is not obj:
				raise KeyError(f"{self._name!s} '{k}' already registered with a different object")
			self._values[k] = obj
			if description:
				self._descriptions[k] = description

	# Public API
	def register(self, key: str | Iterable[str], obj: T, description: str = ""):
		"""Register *obj* under *key*."""
		self._add(key, obj, description)
		return obj

	def get(self, key: str) -> T:
		"""Retrieve an object by *key*.

		:raises KeyError: If the key is unknown.
		"""
		try:
			return self._values[key]
		except KeyError as exc:
			available = ", ".join(sorted(self._values)) or "<empty>"
			raise KeyError(
				f"Unknown {self._name} '{key}'. Available {self._name}s: [{available}]"
			) from exc

	def get_description(self, key: str) -> str:
		"""Retrieve the description of an object by *key*.

		:raises KeyError: If the key is unknown.
		"""
		try:
			return self._descriptions[key]
		except KeyError as exc:
			available = ", ".join(sorted(self._values)) or "<empty>"
			raise KeyError(
				f"Unknown {self._name} '{key}'. Available {self._name}s: [{available}]"
			) from exc

	def available(self) -> List[str]:
		"""Return the list of registered keys (sorted for reproducibility)."""
		return sorted(self._values.keys())

	def __contains__(self, item: str) -> bool:
		return item in self._values

	def __iter__(self):
		return iter(self._values)

	def __len__(self):
		return len(self._values)

	def __repr__(self) -> str:
		n = len(self)
		keys = ", ".join(sorted(self._values))
		return f"<Registry[{self._name}] ({n} items): {keys}>"

	def __getitem__(self, key: str) -> T:
		return self.get(key)


# Global registries exposed by the library
SCHEDULERS: Registry = Registry("scheduler")
PARTITIONERS: Registry = Registry("partitioner")


class SchedulerFn(Protocol):
	"""
	Function that returns a list of operations to be performed.
	"""

	def __call__(
		self, placement: str, n_micro_batches: int, signatures: List[elf.partitioners.utils.Signature]
	) -> List[elf.scheduling.Operation]:
		"""
		:param placement: device on which each block is placed
		:type placement: List[int]
		:param n_micro_batches: number of micro batches
		:type n_micro_batches: int
		:param signatures: signatures representing dependencies between blocks
		:type signatures: List[Signature]

		:return: a list containing the operations to execute for **all** processes
		:rtype: List[Operation]
		"""

	...


# For type checking
Scheduler: TypeAlias = SchedulerFn


class PartitionerFn(Protocol):
	"""
	Function that splits a graph into a list of subgraphs.
	"""

	def __call__(
		self, graph: torch.fx.GraphModule, times: Dict[str, float], memories: Dict[str, float], n: int
	) -> List[List[torch.fx.Node]]:
		"""
		:param graph: Computation graph of the model, in torch.fx format
		:type graph: fx.GraphModule
		:param times: profiled execution time of each node
		:type times: Dict[str, float]
		:param memories: memory size of the output of each node
		:type memories: Dict[str, float]
		:param n: number of partitions to create
		:type n: int

		:return: n lists of nodes corresponding to each part
		:rtype: List[List[fx.Node]]

		"""
		...


Partitioner: TypeAlias = PartitionerFn

__all__ += ["Scheduler", "Partitioner"]
