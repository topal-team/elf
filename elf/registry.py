"""Generic plugin registries used across the **elf** code-base.

End-users can extend the library by simply importing :py:mod:`elf.registry` and calling :py:func:`Registry.register`.

Example
-------
>>> from elf.registry import SCHEDULERS
>>> def custom_scheduler(placement, n_micro_batches, signatures):
...     ...
...     return schedule

>>> SCHEDULERS.register("my_scheduler", custom_scheduler, "My custom scheduler")

Later, inside :py:class:`elf.pipeline.Pipeline` you can refer to it with

>>> pipeline = Pipeline(model, sample, scheduler="my_scheduler")
or
>>> pipeline = Pipeline(model, sample, scheduler=custom_scheduler)
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, Iterable, List, TypeVar, Protocol, TypeAlias

import torch

__all__ = ["Registry", "SCHEDULERS", "PARTITIONERS", "TRACERS", "COMM_SCHEDULERS", "resolve"]

T = TypeVar("T", bound=Callable[..., object])


class Registry(Generic[T]):
	"""Minimal dictionary-backed registry.

	:param name: Name used in error messages (e.g. *"scheduler"*).
	:type name: str
	"""

	def __init__(self, name: str):
		self._name: str = name
		self._values: Dict[str, T] = {}
		self._descriptions: Dict[str, str] = {}

	def _add(self, keys: str | Iterable[str], obj: T, description: str = "") -> None:
		"""Internal primitive that inserts *(key → obj)*, with duplicate-check."""
		if isinstance(keys, str):
			keys = [keys]
		for k in keys:
			existing = self._values.get(k)
			if existing is not None and existing is not obj:
				raise KeyError(f"{self._name!s} '{k}' already registered with a different object")
			self._values[k] = obj
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

	def get_key(self, obj: T) -> str:
		"""Retrieve the key of an object."""
		for key, value in self._values.items():
			if value is obj:
				return key
		raise ValueError(f"Object {obj} not found in {self._name}")

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
		# Deduplicate by tracking seen objects
		seen_objs = set()
		unique_keys = []
		for key in sorted(self._values.keys()):
			obj = self._values[key]
			obj_id = id(obj)
			if obj_id not in seen_objs:
				seen_objs.add(obj_id)
				unique_keys.append(key)
		return unique_keys

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
SCHEDULERS: Registry[SchedulerFn] = Registry("scheduler")
PARTITIONERS: Registry[PartitionerFn] = Registry("partitioner")
TRACERS: Registry[TracerFn] = Registry("tracer")
COMM_SCHEDULERS: Registry[CommSchedulerFn] = Registry("comm_scheduler")


def resolve(var: str | Callable, registry: Registry) -> T:
	"""
	Resolve a name to a callable.
	"""
	if callable(var):
		return var
	elif var in registry:
		return registry[var]
	else:
		msg = f"Unknown {registry._name} '{var}'. Available ones:\n"
		for name in registry.available():
			msg += f"\t{name}: {registry.get_description(name)}\n"
		raise ValueError(msg)


class SchedulerFn(Protocol):
	"""
	Function that returns a list of operations to be performed.
	"""

	def __call__(self, placement: List[int], n_micro_batches: int) -> List[Any]:
		"""
		:param placement: device on which each block is placed (Placement object, which is a List[int])
		:type placement: Placement (List[int])
		:param n_micro_batches: number of micro batches
		:type n_micro_batches: int

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

	@abstractmethod
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


class TracerFn(Protocol):
	"""
	Extract computation graph from a model using torch FX format.
	"""

	@abstractmethod
	def __call__(
		self, model: torch.nn.Module, sample: torch.Tensor, *args, **kwargs
	) -> torch.fx.GraphModule:
		"""
		:param model: Model to trace
		:type model: torch.nn.Module
		:param sample: Sample input to use for tracing, if needed
		:type sample: torch.Tensor

		:return: Computation graph of the model, in torch.fx format
		:rtype: fx.GraphModule
		"""
		...


Tracer: TypeAlias = TracerFn


class CommSchedulerFn(Protocol):
	"""
	Function that reorders communications in a schedule. The order of computations should NOT be changed.
	"""

	def __call__(self, schedule: List[Operation]) -> List[Operation]:  # noqa: F821 #type: ignore
		"""
		:param schedule: Schedule to reorder
		:type schedule: List[Any]

		:return: Reordered schedule
		:rtype: List[Operation]
		"""
		...


CommScheduler: TypeAlias = CommSchedulerFn

__all__ += ["Scheduler", "Partitioner", "Tracer", "CommScheduler"]
