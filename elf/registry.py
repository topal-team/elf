"""Generic plugin registries used across the **elf** code-base.

End-users can extend the library by simply importing :py:mod:`elf.registry` and calling :py:func:`Registry.register`.

Example
-------
>>> from elf.registry import SCHEDULERS
>>> def custom_scheduler(placement, n_micro_batches, profiles):
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

from elf.topology import Placement, true_

import torch

__all__ = ["Registry", "SCHEDULERS", "PARTITIONERS", "TRACERS", "COMM_SCHEDULERS", "resolve"]

T = TypeVar("T", bound=Callable[..., object])


class Registry(Generic[T]):
	"""Minimal dictionary-backed registry.

	:param name: Name used in error messages (e.g. *"scheduler"*).
	:type name: str
	:param adapt: Applied to every object registered or resolved, to bring it to the interface of
		the registry. Defaults to the identity. Adapting the same object twice must give equal
		results, so that registering it again is not seen as a conflict.
	:type adapt: Callable[[Any], T] | None
	"""

	def __init__(self, name: str, adapt: Callable[[Any], T] | None = None):
		self._name: str = name
		self._adapt = adapt
		self._values: Dict[str, T] = {}
		self._descriptions: Dict[str, str] = {}

	def _add(self, keys: str | Iterable[str], obj: T, description: str = "") -> None:
		"""Internal primitive that inserts *(key → obj)*, with duplicate-check."""
		if isinstance(keys, str):
			keys = [keys]
		for k in keys:
			existing = self._values.get(k)
			if existing is not None and existing != obj:
				raise KeyError(f"{self._name!s} '{k}' already registered with a different object")
			self._values[k] = obj
			self._descriptions[k] = description

	# Public API
	def adapt(self, obj: Any) -> T:
		"""Bring *obj* to the interface of the registry, see the ``adapt`` constructor argument."""
		return obj if self._adapt is None else self._adapt(obj)

	def register(self, key: str | Iterable[str], obj: Any, description: str = "") -> T:
		"""Register *obj* under *key*, adapted to the interface of the registry."""
		obj = self.adapt(obj)
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
			if value == obj:
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
		"""Return the list of registered keys for unique values (sorted for reproducibility)."""
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

	def available_keys(self) -> List[str]:
		"""Return the list of all registered keys, including those pointing to duplicate values."""
		return list(self._values.keys())

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


class Scheduler:
	"""
	A scheduling function paired with the check of the placements it supports.

	:param scheduler: function ``(placement, n_micro_batches, profiles) -> List[Operation]``, with
		the arguments and result documented in :meth:`__call__`. It always receives ``profiles``.
	:type scheduler: Callable
	:param checker: function ``(placement, n_micro_batches) -> bool``, see :meth:`check`. Accepts
		everything by default.
	:type checker: Callable
	"""

	def __init__(self, scheduler: Callable, checker: Callable = true_):
		self.scheduler = scheduler
		self.checker = checker

	def __call__(
		self,
		placement: Placement,
		n_micro_batches: int,
		profiles: List[StageProfile] | None = None,  # noqa: F821
	) -> List[Any]:
		"""
		Generate the schedule.

		:param placement: device on which each block is placed (Placement object, which is a List[int])
		:type placement: Placement (List[int])
		:param n_micro_batches: number of micro batches
		:type n_micro_batches: int
		:param profiles: one profile per stage, or None when the pipeline did not profile the model
		:type profiles: List[StageProfile] | None

		:return: a list containing the operations to execute for **all** processes
		:rtype: List[Operation]
		"""
		return self.scheduler(placement, n_micro_batches, profiles)

	def check(self, placement: Placement, n_micro_batches: int) -> bool:
		"""
		Check that the placement and number of microbatches are valid for this scheduler. This is a
		guard against invalid combinations, e.g. 1-wave Hanayo.

		:rtype: bool
		"""
		return self.checker(placement, n_micro_batches)

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, Scheduler):
			return NotImplemented
		return (self.scheduler, self.checker) == (other.scheduler, other.checker)

	def __hash__(self) -> int:
		return hash((self.scheduler, self.checker))


def _as_scheduler(obj: Any) -> Scheduler:
	"""Wrap a bare scheduling function in a :class:`Scheduler` that accepts every placement."""
	return obj if hasattr(obj, "check") else Scheduler(obj)


# Global registries exposed by the library
SCHEDULERS: Registry[Scheduler] = Registry("scheduler", adapt=_as_scheduler)
PARTITIONERS: Registry[PartitionerFn] = Registry("partitioner")
TRACERS: Registry[TracerFn] = Registry("tracer")
COMM_SCHEDULERS: Registry[CommSchedulerFn] = Registry("comm_scheduler")


def resolve(var: str | Callable, registry: Registry) -> T:
	"""
	Resolve a name to a callable. A callable is adapted to the interface of the registry, like a
	registered one.
	"""
	if callable(var):
		return registry.adapt(var)
	elif var in registry:
		return registry[var]
	else:
		msg = f"Unknown {registry._name} '{var}'. Available ones:\n"
		for name in registry.available():
			msg += f"\t{name}: {registry.get_description(name)}\n"
		raise ValueError(msg)


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

	def __call__(  # noqa: F821 #type: ignore
		self,
		schedule: List[Operation],  # noqa: F821
		profiles: List[StageProfile] | None = None,  # noqa: F821
	) -> List[Operation]:  # noqa: F821
		"""
		:param schedule: Schedule to reorder
		:type schedule: List[Any]
		:param profiles: One :py:class:`~elf.profiling.StageProfile` per stage, or None when the
			pipeline was built without profiling. A scheduler that only reasons about the order of
			the operations ignores them.
		:type profiles: List[StageProfile] or None

		:return: Reordered schedule
		:rtype: List[Operation]
		"""
		...


CommScheduler: TypeAlias = CommSchedulerFn

__all__ += ["Scheduler", "Partitioner", "Tracer", "CommScheduler"]
