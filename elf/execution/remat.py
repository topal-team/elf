"""
Rematerialization engine.

The forward runs with gradients enabled, so the graph is created, but every tensor the graph saves is replaced by a :class:`Holder`.
Emptying the holders frees the activations while leaving the graph itself intact. Filling them again makes it usable.

Every activation of a recomputed unit, including its output, is deleted, no matter where else it is saved.
The one exception is an output that another unit keeps as its own input: it is in memory anyway, so
it is pinned, and whoever else saved it reads it from there instead of rebuilding it.

There are exactly two ways to fill a unit back up, and which one applies follows from what was
freed: if the inputs of the recomputed region are still live the units are re-run, and if they are
not the whole model forward is replayed from the stage inputs.

Units are placed with forward hooks on the modules a strategy selects, so the model is never
modified and the order the units are called in is discovered rather than declared.
"""

import logging
import weakref

from contextlib import ExitStack, contextmanager
from functools import partial

import torch

from elf.scheduling import RematMode
from elf.utils import rng_state, restored_rng, tensor_leaves

__all__ = ["Holder", "Unit", "Stage", "unit_placement"]

logger = logging.getLogger(__name__)


def unit_placement(remat_strategy=None, replayable=False):
	"""
	Decide which modules become units.

	A unit is the granularity at which activations are dropped and rebuilt, and at which they are
	released again as a backward walks through them:

	- forward rematerialization puts a unit on every module ``remat_strategy`` gives a mode to;
	- backward rematerialization has to be able to rebuild the whole stage after ``B``, so every
	  module the strategy leaves alone becomes a unit that keeps its activations until they are
	  dropped. Covering the stage in pieces rather than in one is what lets each piece go as soon
	  as its own backward is over, which matters because that backward retains the graph and
	  autograd will not release anything by itself.

	The recomputed modules end up nested inside the kept ones, which is allowed: a kept unit is
	only ever rebuilt by a replay, and a replay reinstates every unit. The converse is not, so a
	module inside a recomputed one does not become a unit at all -- recomputing a module re-runs it
	as a whole.

	:param remat_strategy: takes a module name and module and returns its mode, or None to leave
		it alone, as carried by ``OpOptions.REMAT_STRATEGY``
	:type remat_strategy: Callable[[str, torch.nn.Module], RematMode or None] or None
	:param replayable: whether the schedule drops this stage's activations between ``B`` and ``W``
	:type replayable: bool
	:return: a placement strategy for :class:`Stage`
	:rtype: Callable[[str, torch.nn.Module], RematMode or None]
	"""

	def placement(name, module):
		mode = remat_strategy(name, module) if remat_strategy is not None else None
		if mode is not None:
			return mode
		if replayable:
			return RematMode.KEEP
		return None

	return placement


# === Utils for nested structures, not useful in most cases ===


def _is_inside(name, ancestor):
	"""Whether a module is inside another, the root module being named ""."""
	return ancestor == "" or name.startswith(ancestor + ".")


def _identity(tensor):
	return tensor


def _rebuilt(container, items):
	"""
	Rebuild a container of the same type from the items of a walk over it.

	A named tuple takes its fields positionally rather than as one iterable, and rebuilding it with
	the iterable would silently produce a one-field tuple.
	"""
	if isinstance(container, tuple) and hasattr(container, "_fields"):
		return type(container)(*items)
	return type(container)(items)


_TENSOR = object()  # marks where a tensor sat in a structure it was taken out of


def _is_tensor(value):
	return isinstance(value, torch.Tensor)


def _mapped(value, leaf, is_leaf=_is_tensor):
	"""
	Rebuild a structure of tuples, lists and dicts, with ``leaf`` applied to each of its leaves.

	The leaves are visited in the order :func:`~elf.utils.tensor_leaves` walks them, so a mapping
	that draws from an iterator lines up with a flat recording of the same structure.

	:param leaf: what to put in place of each leaf
	:param is_leaf: what counts as a leaf, tensors by default
	"""
	if is_leaf(value):
		return leaf(value)
	if isinstance(value, (tuple, list)):
		return _rebuilt(value, [_mapped(item, leaf, is_leaf) for item in value])
	if isinstance(value, dict):
		return {key: _mapped(item, leaf, is_leaf) for key, item in value.items()}
	return value


def _detached(value):
	return _mapped(value, lambda tensor: tensor.detach())


def _hollowed(args):
	"""The input structure with all tensors replaced by dummy objects."""
	return _mapped(args, lambda _: _TENSOR)


def _filled(template, leaves):
	"""Put tensors back where :func:`_hollowed` took them out, in the order it walked them."""
	return _mapped(template, lambda _: next(leaves), is_leaf=lambda value: value is _TENSOR)


def _with_grad_flags(args, flags):
	"""
	Rebuild an input structure with the ``requires_grad`` flags of the original forward.

	A replay runs with gradients disabled, and so does the tail of a recomputation handing its
	results to the next unit, so intermediate results arrive at a unit not requiring grad. Without
	this the unit would save a different set of tensors than the forward did, and its holders would
	no longer line up with them.

	A tensor that already carries the flag it had is left exactly as it is: inside a unit that is
	being recomputed, the results flowing into a nested one belong to the recomputation's graph,
	and detaching them would cut the outer unit off from what it is meant to save.

	:param args: the structure a replay produced
	:param flags: iterator over the flags recorded by :meth:`Unit.bind`
	:return: the same structure, with the flags restored
	"""

	def restore(tensor):
		wanted = next(flags)
		if wanted and not tensor.requires_grad:
			return tensor.detach().requires_grad_(True)
		return tensor

	return _mapped(args, restore)


# === Utils to identify tensors by storage to handle views and aliases ===


def _span_of(size, stride):
	"""How far into a storage a shape and its strides reach, in elements."""
	if 0 in size:
		return 0
	return 1 + sum((extent - 1) * step for extent, step in zip(size, stride))


def _span(tensor):
	"""How far into its storage a tensor reaches, in elements."""
	return _span_of(tuple(tensor.shape), tuple(tensor.stride()))


def _viewed(base, view):
	"""Read what was saved out of ``base``, at the place :func:`_view_of` found it."""
	size, stride, delta = view
	return base.as_strided(size, stride, base.storage_offset() + delta)


def _geometry(tensor):
	"""Where a tensor sits in its storage, for :func:`_at` to find it again."""
	return tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset()


def _at(tensor, geometry):
	"""A detached tensor at ``geometry`` in the storage of ``tensor``."""
	size, stride, offset = geometry
	return tensor.detach().as_strided(size, stride, offset)


def _view_of(base, saved):
	"""
	How to rebuild a tensor the graph saved out of the tensor a unit returned.

	What a node saves is rarely the tensor it was handed: a linear layer applied to a batch of
	sequences saves it flattened, and a slice or a transpose saves a view just the same. All of
	them read the same storage, so the unit that produced it can produce them again.

	:return: the shape, the strides and how far into the unit's tensor it starts, or None if it is
		not a view of it at all, in which case whoever saved it has to keep it
	:rtype: tuple or None
	"""
	if saved.dtype is not base.dtype:
		return None
	if saved.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
		return None

	delta = saved.storage_offset() - base.storage_offset()
	if delta < 0 or delta + _span(saved) > _span(base):
		return None
	return tuple(saved.shape), tuple(saved.stride()), delta


class _Stop(BaseException):
	"""
	Raised out of a recomputation that has saved everything it was run for.

	A BaseException, so that a module catching what goes wrong in its own forward does not catch
	this as well.
	"""


class Holder:
	"""
	What an autograd node stores in place of a saved tensor.

	The value lives here rather than on the unit, so that a graph which is not retained releases
	its activations on its own as it is walked, exactly as it would without this engine.
	"""

	__slots__ = ("unit", "value", "__weakref__")

	def __init__(self, unit, value):
		self.unit = unit
		self.value = value


class Unit:
	"""
	One call of one selected module, and the tensors the graph saved during it.

	A module called several times in a pass produces one unit per call.
	"""

	def __init__(self, module, name, index, mode=RematMode.KEEP):
		"""
		:param index: position of this unit in the order the forward entered them
		:type index: int
		:param mode: what the strategy asked of the module
		:type mode: RematMode
		"""
		self.module = module
		self.name = name
		self.index = index
		self.mode = mode
		self.holders = []  # weak references, in the order the forward saved them
		self.output_holders = {}  # position of an output -> the holders standing for it elsewhere
		self.inputs = None
		self.producers = None  # (unit, position) per input tensor, when the inputs were not kept
		self.template = None  # the shape of the inputs, to put what the producers give back into
		self.outputs = None  # what a recomputation produced, while a unit reading it still needs it
		self.is_producer = False
		self.pins = {}  # position of an output -> (the tensor keeping it alive, where it sits in it)
		self.grad_flags = []
		self.rng = None
		self.n_recomputes = 0
		self._stack = None
		self._buffer = None
		self._saved_by = None
		self._stop_early = False

	@property
	def recomputes(self):
		"""
		Whether this unit keeps nothing from its forward, and recomputes it when asked.

		:rtype: bool
		"""
		return self.mode is not RematMode.KEEP

	def bind(self, args, kwargs, leaves, producers=None):
		"""
		Record what this unit was called with.

		A unit that keeps nothing from its forward has to be able to run again, which means holding
		on to what it was called with. A unit that discards its inputs lets go of them when they are
		the outputs of recomputed units, and records where they come from instead, to ask for them
		back when it needs them.

		:param leaves: the input tensors, in the order a walk over the arguments finds them
		:type leaves: list[torch.Tensor]
		:param producers: the unit that produced each of the input tensors and the position it
			produced it at, or None when the unit keeps its inputs
		:type producers: list[tuple[Unit, int]] or None
		"""
		self.grad_flags = [tensor.requires_grad for tensor in leaves]
		self.rng = rng_state(leaves) if self.recomputes else None

		self.producers = producers if self.recomputes else None
		self.inputs = (args, kwargs) if self.recomputes and self.producers is None else None
		self.template = _hollowed((args, kwargs)) if self.producers else None
		for producer, _ in self.producers or []:
			producer.is_producer = True

	def can_refill(self):
		"""
		Whether this unit could fill itself again, on its own or by asking the units ahead of it.

		:rtype: bool
		"""
		return self.inputs is not None or bool(self.producers)

	def rebind(self, args, kwargs, retain=True):
		"""
		Bind the inputs a replay produced, restoring the flags of the original forward.

		What is kept is a detached copy rather than the inputs themselves.

		:param retain: whether this unit is being filled, and so may have to re-run on its own
			later. A unit the replay only passes through has no use for its inputs.
		:type retain: bool
		:return: the inputs to run the module on
		:rtype: tuple
		"""
		inputs = _with_grad_flags((args, kwargs), iter(self.grad_flags))
		if self.recomputes and retain and not self.producers:
			self.inputs = _with_grad_flags(_detached(inputs), iter(self.grad_flags))
		return inputs

	def begin(self, saved_by=None):
		"""
		Record into the live graph: saved tensors are replaced by holders.

		:param saved_by: what the stage uses to tell whether a tensor being saved was produced by a
			recomputed unit, and so belongs to that unit rather than to this one
		:type saved_by: Callable[[torch.Tensor], tuple[Unit, int, tuple] or None] or None
		"""
		self.holders = []
		self.output_holders = {}
		self._saved_by = None if self.recomputes else saved_by
		self._stack = ExitStack()
		self._stack.enter_context(torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack))

	def end(self):
		self._close()

	def abort(self):
		"""
		Give up on a recording a forward left open by raising, so that the hooks it installed do
		not outlive it, and drop what it had collected so far.
		"""
		if self._stack is not None:
			self._close()

	def _close(self):
		"""
		Take down whatever a ``begin`` put up.

		:return: what a recording collected, and nothing if this was not one
		:rtype: list or None
		"""
		self._stack.close()
		self._stack = None
		self._saved_by = None
		buffer, self._buffer = self._buffer, None
		return buffer

	def hold_output(self, position, view):
		"""
		A slot for a tensor this unit produced, saved by a node outside of it.

		Recomputing the unit is what fills it, the same way it fills the tensors saved inside it.

		:param position: which of the tensors the unit returned, in the order they are walked
		:type position: int
		:param view: how to read what was saved out of that tensor, as :func:`_view_of` gives it
		:type view: tuple
		:rtype: Holder
		"""
		holder = Holder(self, None)
		pinned = self.pinned(position)
		if pinned is not None:
			holder.value = _viewed(pinned, view)
		self.output_holders.setdefault(position, []).append((weakref.ref(holder), view))
		return holder

	def pin(self, position, keeper, base):
		"""
		Keep one of the outputs for as long as another unit keeps it as its input.

		Rebuilding a tensor that is in memory anyway would re-run this unit for nothing, so the
		nodes that saved it read it from there instead, and so does a unit asking for it back.
		Only the keeper is referenced, and weakly: the pin lasts exactly as long as the unit keeping
		it holds on to its inputs.

		:param position: which of the tensors the unit returned
		:type position: int
		:param keeper: what the other unit keeps, sharing the storage of that output
		:type keeper: torch.Tensor
		:param base: the output itself
		:type base: torch.Tensor
		"""
		alive = [entry for entry in self.pins.get(position, ()) if entry[0]() is not None]
		self.pins[position] = alive + [(weakref.ref(keeper), _geometry(base))]

		value = self.pinned(position)
		for reference, view in self.output_holders.get(position, ()):
			holder = reference()
			if holder is not None and holder.value is None:
				holder.value = _viewed(value, view)

	def pinned(self, position):
		"""
		The output at ``position``, if another unit still keeps it as its input.

		:rtype: torch.Tensor or None
		"""
		for reference, geometry in self.pins.get(position, ()):
			keeper = reference()
			if keeper is not None:
				return _at(keeper, geometry)
		return None

	def fill_outputs(self, leaves):
		"""
		Give the outputs produced by this unit to the nodes that saved them

		:param leaves: what the unit returned, walked in order and free of the graph that made it
		:type leaves: list[torch.Tensor]
		"""
		for position, held in self.output_holders.items():
			if position >= len(leaves):
				raise RuntimeError(
					f"{self.name}: recomputing returned {len(leaves)} tensors where the forward "
					f"returned at least {position + 1}. A recomputation has to reproduce the "
					"forward it replaces."
				)

			base = leaves[position]
			for reference, view in held:
				holder = reference()
				# a holder that is still filled was pinned, and refilling it would make a copy
				if holder is None or holder.value is not None:
					continue
				size, stride, delta = view
				if delta + _span_of(size, stride) > _span(base):
					raise RuntimeError(
						f"{self.name}: recomputing produced a tensor of {tuple(base.shape)} that "
						f"what was saved out of it, of {size}, does not fit in. A recomputation "
						"has to reproduce the forward it replaces."
					)
				holder.value = _viewed(base, view)

	def begin_capture(self, restore_rng=True, saved_by=None, stop_early=False):
		"""
		Recompute into an existing graph: the saved tensors are collected in the order they are
		saved and the recomputation's own graph is thrown away.

		:param restore_rng: put the generators back where this unit's forward found them. A unit
			re-run on its own needs that, since it draws out of the order the forward drew in. A
			replay does not, and must not: restoring is a fork, which leaves the generator where it
			was, and the units the replay runs without filling would then draw a state the forward
			had already moved past.
		:type restore_rng: bool
		:param saved_by: see :meth:`begin`
		:param stop_early: raise :class:`_Stop` as soon as the last of this unit's tensors is
			saved, leaving the rest of the forward unrun
		:type stop_early: bool
		"""
		self._buffer = []
		self._stop_early = stop_early
		self._saved_by = None if self.recomputes else saved_by
		self._stack = ExitStack()
		self._stack.enter_context(torch.enable_grad())
		if restore_rng:
			self._stack.enter_context(restored_rng(self.rng))
		self._stack.enter_context(
			torch.autograd.graph.saved_tensors_hooks(self._pack_capture, _identity)
		)

	def end_capture(self):
		buffer = self._close()

		if len(buffer) != len(self.holders):
			raise RuntimeError(
				f"{self.name}: recomputing saved {len(buffer)} tensors where the forward saved "
				f"{len(self.holders)}. A recomputation has to reproduce the forward it replaces, "
				"so the module must be deterministic and must not depend on anything outside its "
				"arguments."
			)

		for reference, value in zip(self.holders, buffer):
			holder = reference()
			if holder is not None:
				holder.value = value
		self.n_recomputes += 1

	def _pack(self, tensor):
		if self._saved_by is not None:
			produced = self._saved_by(tensor)
			if produced is not None:
				unit, position, view = produced
				return unit.hold_output(position, view)

		# a unit that recomputes itself keeps nothing from its forward
		holder = Holder(self, None if self.recomputes else tensor)
		self.holders.append(weakref.ref(holder))
		return holder

	def _unpack(self, holder):
		if holder.value is None:
			holder.unit.recompute()
		return holder.value

	def _pack_capture(self, tensor):
		# The recomputation's own graph is thrown away, but a tensor it saved would point back at
		# it through grad_fn, and a cycle through grad_fn keeps cpp objects the collector cannot
		# free. Handing it a detached alias leaves nothing pointing back.
		value = tensor.detach()
		if self._saved_by is None or self._saved_by(tensor) is None:
			self._buffer.append(value)  # what belongs to another unit was never in this list
			if self._stop_early and len(self._buffer) == len(self.holders):
				raise _Stop
		return value

	def recompute(self):
		"""
		Fill this unit again, running whatever its inputs come from first.

		A unit that let go of its inputs gets them back from the units that produced them, which
		have to run first, and those from theirs; the cascade stops at the first unit that still
		holds what it was called with, or whose output another unit has pinned. Every unit it
		passes through is filled on the way and keeps what it produced.

		:raises RuntimeError: if a unit it needs can neither run nor say where its inputs come
			from, which only a replay can undo
		"""
		pending = {}
		stack = [self]
		while stack:
			unit = stack.pop()
			if unit.index in pending:
				continue

			pending[unit.index] = unit
			if unit.inputs is not None:
				continue
			if not unit.producers:
				raise RuntimeError(
					f"{unit.name}: its inputs were dropped, so it cannot be recomputed on its own. "
					"Replay the stage instead."
				)

			# a unit that already produced what it is being asked for does not run twice, which is
			# what keeps a tensor several units read from being rebuilt once per reader
			stack.extend(
				producer
				for producer, position in unit.producers
				if producer.outputs is None and producer.pinned(position) is None
			)

		for index in sorted(pending):  # a producer always comes before what reads it
			pending[index]._rerun()

	def _rerun(self):
		"""
		Run the module over the saved tensors of the graph, and keep what it produced.

		When nothing needs what it produces, the module stops as soon as it has saved its last
		tensor: whatever it computes after that is thrown away, and is not run at all.
		"""
		args, kwargs = self._arguments()
		self.begin_capture(stop_early=not self._outputs_wanted())
		try:
			output = self.module(*args, **kwargs)
		except _Stop:
			self.end_capture()
			return
		except BaseException:
			# clear the buffer and let the real exception through
			self.abort()
			raise
		self.end_capture()

		# detached, because the graph this recomputation built dies with it, and what reads these
		# -- a node that saved one of them, or a unit recomputing itself next -- is not part of it
		leaves = [tensor.detach() for tensor in tensor_leaves(output)]
		self.fill_outputs(leaves)
		if self.is_producer:
			self.outputs = leaves

	def _outputs_wanted(self):
		"""
		Whether a re-run of this unit has to produce its outputs, rather than only its saved tensors.

		It does when a unit that discarded its inputs may ask for them, or when a node outside the
		unit saved one of them and is still waiting for it. A pinned output is already filled.

		:rtype: bool
		"""
		if self.is_producer:
			return True
		for held in self.output_holders.values():
			for reference, _ in held:
				holder = reference()
				if holder is not None and holder.value is None:
					return True
		return False

	def _arguments(self):
		"""
		What to run the module on: what it was called with, or what its producers just made.

		:rtype: tuple
		"""
		if self.inputs is not None:
			return self.inputs

		leaves = []
		for producer, position in self.producers:
			if producer.outputs is not None and position < len(producer.outputs):
				leaves.append(producer.outputs[position])
				continue

			pinned = producer.pinned(position)
			if pinned is None:
				raise RuntimeError(
					f"{self.name}: {producer.name} did not produce the tensor it produced the first "
					"time round, so there is nothing to hand back. A recomputation has to reproduce "
					"the forward it replaces."
				)
			leaves.append(pinned)

		return _with_grad_flags(_filled(self.template, iter(leaves)), iter(self.grad_flags))

	def _live(self):
		"""
		The holders this unit placed in the graph that the graph is still using.

		A holder the graph has let go of is one a backward walked through without retaining it:
		what it carried is gone for good, and nobody can ask for it again.

		:rtype: Iterator[Holder]
		"""
		for reference in self.holders:
			holder = reference()
			if holder is not None:
				yield holder
		for held in self.output_holders.values():
			for reference, _ in held:
				holder = reference()
				if holder is not None:
					yield holder

	def needs_filling(self):
		"""
		Whether this unit is missing activations something could still ask it for.

		:rtype: bool
		"""
		return any(holder.value is None for holder in self._live())

	def free(self):
		"""
		Empty the holders, and let go of whatever nothing can ask for any more.

		What the unit was called with goes too, once the graph has let go of every holder it
		filled. A unit with no holder left in the graph is one nothing can unpack, so nothing can
		ask it to run again.

		An output another unit still keeps as its input stays: it is in memory anyway, and
		emptying it would only mean rebuilding it.
		"""
		for reference in self.holders:
			holder = reference()
			if holder is not None:
				holder.value = None
		for position, held in self.output_holders.items():
			if self.pinned(position) is not None:
				continue
			for reference, _ in held:
				holder = reference()
				if holder is not None:
					holder.value = None
		# whatever reads these has had its own backward by now, so nothing is waiting for them
		self.outputs = None
		if next(self._live(), None) is None:
			self.inputs = None

	def drop(self):
		"""
		Empty the holders, pinned ones included, and let go of the inputs. Only a replay can
		rebuild the unit.
		"""
		for holder in self._live():
			holder.value = None
		self.outputs = None
		self.inputs = None
		self.pins = {}

	def values(self):
		"""
		The activations this unit is currently holding.

		:return: the live saved tensors
		:rtype: Iterator[torch.Tensor]
		"""
		for holder in self._live():
			if holder.value is not None:
				yield holder.value


class Stage:
	"""
	One forward of a model, and the units it went through, in the order they were called.
	"""

	def __init__(self, model, strategy, replayable=True):
		"""
		:param strategy: takes a module name and module, and returns the mode of the unit to place
			on it, or None for a module that should not be a unit at all
		:type strategy: Callable[[str, torch.nn.Module], RematMode or None]
		:param replayable: keep the stage inputs, so that the whole forward can be replayed. Only a
			schedule that drops the activations between ``B`` and ``W`` needs that; without it a
			block is free to let go of its own inputs.
		:type replayable: bool
		"""
		self.model = model
		self.strategy = strategy
		self.replayable = replayable
		self.units = []
		self.roots = None
		self.rng = None
		self._open = []  # the units the forward is inside of, innermost last
		self._skipped = 0  # how deep inside a recomputed unit it is, where nothing is a unit
		self._capture = None  # the units a replay is filling, None when this is not a replay
		self._filled = 0  # how many units that replay has reached
		self._produced = {}  # storage -> the unit that produced it, for the length of one forward
		self._recomputing = False  # whether any unit recomputes, and so deletes what it produces
		self._selection = None  # which modules are units, settled once for the life of the stage

	def run(self, *args, **kwargs):
		"""
		The ``F`` operation: run the model, placing a unit on every selected module.

		:return: whatever the model returns
		"""
		self.units = []
		self.roots = (args, kwargs) if self.replayable else None
		self.rng = rng_state((args, kwargs))
		self._capture = None
		self._produced = {}

		with ExitStack() as stack:
			stack.enter_context(self._hooks())
			if self._recomputing:
				# a tensor a recomputed unit produced has to be caught wherever it is saved, and
				# most of the modules that save one are not units themselves
				stack.enter_context(
					torch.autograd.graph.saved_tensors_hooks(self._pack_outside, self._unpack)
				)
			return self.model(*args, **kwargs)

	def replay(self, capture=None):
		"""
		The ``RF`` operation: run the model again from the stage inputs and fill the units back up.

		Only the units being filled run with gradients enabled, and the outputs are detached, so no additional
		graph is kept from the recomputation. The activations it fills stay, of course: they are what it ran for.

		A replay costs what the stage is actually missing: a unit that still has its activations is not recomputed, and neither is
		one whose holders a backward has already consumed, since nothing will read those again.

		:param capture: the units to fill, by default the ones that are missing something. The
			others still run, since their results are needed downstream, but retain nothing.
		:type capture: Iterable[Unit] or None
		:raises RuntimeError: if the stage was built without keeping its inputs
		"""
		if self.roots is None:
			raise RuntimeError(
				"this stage was built with replayable=False, so its inputs were not kept and the "
				"forward cannot be replayed"
			)

		if capture is None:
			capture = [unit for unit in self.units if unit.needs_filling()]
		self._capture = set(capture)
		self._filled = 0
		self._produced = {}
		args, kwargs = self.roots
		try:
			with torch.no_grad(), restored_rng(self.rng), self._hooks():
				self.model(*args, **kwargs)
		finally:
			self._capture = None

	def drop(self):
		"""
		Free every activation the stage is holding, inputs included.
		"""
		for unit in self.units:
			unit.drop()

	def forget(self):
		"""
		Let go of the inputs a replay would have needed, once no replay can follow.

		The activations themselves are left to the graph, which releases them as a backward that
		does not retain it walks through.
		"""
		self.roots = None
		for unit in self.units:
			unit.inputs = None

	def values(self):
		"""
		The activations the stage is currently holding.

		:rtype: Iterator[torch.Tensor]
		"""
		for unit in self.units:
			yield from unit.values()

	def _selected(self):
		"""
		Which modules are units, and which of them recompute themselves.

		Neither the model nor the strategy changes over the life of a stage, so this is settled
		once and read again by every replay.

		:rtype: list[tuple[str, torch.nn.Module, bool]]
		"""
		if self._selection is not None:
			return self._selection

		found = [
			(name, module, self.strategy(name, module)) for name, module in self.model.named_modules()
		]
		found = [(name, module, mode) for name, module, mode in found if mode is not None]

		for name, _, mode in found:
			if not isinstance(mode, RematMode):
				raise TypeError(
					f"the strategy returned {mode!r} for {name or '<stage>'}, where it has to return "
					"a RematMode, or None to leave the module alone"
				)

		for outer, _, mode in found:
			if mode is RematMode.KEEP:
				continue  # only a unit that re-runs its own module has to be free of units
			for inner, _, inner_mode in found:
				if inner_mode is not RematMode.KEEP and inner != outer and _is_inside(inner, outer):
					raise ValueError(
						f"{inner} is inside {outer}, which recomputes itself. Recomputing a module "
						"does not reinstate the units within it, so a recomputed unit has to be "
						"free of them."
					)

		self._selection = found
		return found

	@contextmanager
	def _hooks(self):
		selected = self._selected()  # before any hook is placed, so that a refusal places none
		handles = []
		self._open = []
		self._skipped = 0
		self._recomputing = any(mode is not RematMode.KEEP for _, _, mode in selected)
		for name, module, mode in selected:
			handles.append(
				module.register_forward_pre_hook(partial(self._pre, name=name, mode=mode), with_kwargs=True)
			)
			handles.append(module.register_forward_hook(self._post, with_kwargs=True))
		try:
			yield
		finally:
			for handle in handles:
				handle.remove()
			# a forward that raised left the units it was inside of open, and their recordings
			# install hooks that have no business outliving this pass
			for unit in self._open:
				unit.abort()
			self._open = []
			self._skipped = 0

	def _pre(self, module, args, kwargs, name, mode):
		# recomputing a module re-runs it as a whole, so what is inside it is not a unit. Only the
		# innermost open unit can be a recomputed one, since nothing inside one is ever opened.
		if self._skipped or (self._open and self._open[-1].recomputes):
			self._skipped += 1
			return None

		if self._capture is None:
			unit = Unit(module, f"{name or '<stage>'}#{len(self.units)}", len(self.units), mode)
			self.units.append(unit)
			leaves = list(tensor_leaves((args, kwargs)))
			discards = mode is RematMode.DISCARD
			unit.bind(args, kwargs, leaves, producers=self._producers_of(leaves) if discards else None)
			self._pin_inputs(unit)
			self._arm(unit, leaves)
			unit.begin(saved_by=self._saved_by if self._recomputing else None)
			self._open.append(unit)
			return None

		index = self._filled
		unit = self.units[index] if index < len(self.units) else None
		if unit is None or unit.module is not module:
			expected = f"{unit.name} to be entered" if unit is not None else "the forward to be over"
			raise RuntimeError(
				f"a replay of this stage reached {name or '<stage>'} where it expected {expected}. "
				"A replay has to reproduce the forward it fills, so the model must take the same "
				"path through its modules as it did then."
			)

		self._filled += 1
		capture = unit in self._capture
		args, kwargs = unit.rebind(args, kwargs, retain=capture)
		self._pin_inputs(unit)
		if capture:
			# the replay runs the units in the order the forward ran them, so the generator is
			# already where this unit left it and restoring would only hold it back
			unit.begin_capture(restore_rng=False, saved_by=self._saved_by if self._recomputing else None)
		self._open.append(unit)
		return args, kwargs

	def _post(self, module, args, kwargs, output):
		if self._skipped:
			self._skipped -= 1
			return None

		unit = self._open.pop()
		if self._capture is None:
			unit.end()
			self._register(unit, output)
			return None

		if unit in self._capture:
			unit.end_capture()
			# the recomputation's graph must not outlive the replay, but only the outermost unit
			# being recomputed can cut it: detaching inside one would sever what its parent is
			# recomputing, and the parent would save nothing
			nested = any(open_unit in self._capture for open_unit in self._open)
			result = output if nested else _detached(output)
			unit.fill_outputs(list(tensor_leaves(result)))
			self._register(unit, result)
			return None if nested else result

		self._register(unit, output)
		return None

	def _register(self, unit, output):
		"""
		Remember which unit produced each of the tensors it returned.

		What a recomputed unit returns is going to be deleted and built again, so a node that saves
		one of those tensors is given a holder belonging to the unit that made it, and a unit that
		discards its inputs has no reason to hold on to it. Only the identity is kept, never the
		tensor.
		"""
		if not unit.recomputes:
			return

		for position, tensor in enumerate(tensor_leaves(output)):
			self._produced[tensor.untyped_storage().data_ptr()] = (weakref.ref(tensor), unit, position)

	def _pack_outside(self, tensor):
		"""
		What the graph saves while no unit is recording.

		A tensor a recomputed unit produced is one this stage deletes, wherever it was saved, and
		the unit that made it is what fills it back in. Everything else is left exactly as it was,
		so a stage pays nothing for what it is not rematerializing.
		"""
		produced = self._saved_by(tensor)
		if produced is None:
			return tensor

		unit, position, view = produced
		return unit.hold_output(position, view)

	def _unpack(self, packed):
		if not isinstance(packed, Holder):
			return packed
		if packed.value is None:
			packed.unit.recompute()
		return packed.value

	def _produced_by(self, tensor):
		"""
		The unit that returned a tensor, when a recomputed one returned this very tensor.

		:rtype: tuple[Unit, int] or None
		"""
		found = self._produced.get(tensor.untyped_storage().data_ptr())
		if found is None:
			return None

		reference, unit, position = found
		if reference() is not tensor:  # a view of it, or a dead tensor whose storage was reused
			return None
		return unit, position

	def _saved_by(self, tensor):
		"""
		The unit a tensor the graph is about to save belongs to, when a recomputed one produced it.

		:rtype: tuple[Unit, int, tuple] or None
		"""
		found = self._produced.get(tensor.untyped_storage().data_ptr())
		if found is None:
			return None

		reference, unit, position = found
		base = reference()
		if base is None:  # gone, so its storage was handed out to somebody else
			return None

		view = _view_of(base, tensor)
		return None if view is None else (unit, position, view)

	def _pin_inputs(self, unit):
		"""
		Pin the outputs of recomputed units that a unit keeps as its own inputs.

		Matching by storage rather than by identity, so that a view of an output pins it as well:
		the unit keeping the view keeps the storage alive all the same.
		"""
		if unit.inputs is None:
			return

		for tensor in tensor_leaves(unit.inputs):
			found = self._produced.get(tensor.untyped_storage().data_ptr())
			if found is None:
				continue
			reference, producer, position = found
			base = reference()
			if base is not None and _view_of(base, tensor) is not None:
				producer.pin(position, tensor, base)

	def _producers_of(self, leaves):
		"""
		Which unit produced each tensor of an input structure.

		:return: the producer and the position it produced it at, per input tensor, or None if any
			of them is not exactly the output of a recomputed unit -- a tensor that went through
			anything on the way, an addition over a skip connection say, is nobody's output and has
			to be kept by whoever reads it
		:rtype: list[tuple[Unit, int]] or None
		"""
		found = []
		for tensor in leaves:
			produced = self._produced_by(tensor)
			if produced is None:
				return None
			found.append(produced)

		return found or None  # a unit called on no tensor at all has nothing to ask anybody for

	def _arm(self, unit, leaves):
		"""
		Arrange for a unit to let go of its activations as soon as its own backward is over.

		A node runs only once everything downstream of it has run, so a hook on the node that
		produced a unit's input fires exactly then. Without it, a backward that retains the graph
		would fill every unit back up and peak at the whole stage rather than at one unit.
		"""
		if not unit.can_refill() and not self.replayable:
			return  # nothing could fill it back up, so its tensors have to stay

		# the node holds the hook and the unit holds its inputs, which hold the node: a strong
		# reference here would close a cycle through a cpp object, which the collector cannot break
		reference = weakref.ref(unit)

		def release(*_):
			released = reference()
			if released is not None:
				released.free()

		armed = False
		for tensor in leaves:
			if tensor.grad_fn is not None:
				tensor.grad_fn.register_hook(release)
				armed = True
			elif tensor.requires_grad:
				tensor.register_post_accumulate_grad_hook(release)
				armed = True

		if not armed:
			# nothing upstream of this unit differentiates -- a stage fed integer token ids, say --
			# so there is no node whose turn marks the end of its backward
			logger.debug(
				f"{unit.name}: none of its inputs carries a gradient, so it holds its activations "
				"until the graph itself is released"
			)
