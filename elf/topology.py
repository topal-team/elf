"""
Everything related to process topology, communicators etc.

The module is in three layers. :class:`Placement` and the functions after it are pure: given a
world size and a placement they say which ranks belong together, and which blocks talk to each
other. :class:`PipelineTopology` sits on top and turns those answers into the actual
``torch.distributed`` process groups, which it then owns for the lifetime of the pipeline. Last,
:func:`send_models` and friends move whole models over those groups, which is how a partition
computed on one rank reaches the others and how the pipeline is gathered back for checkpointing.
"""

import os
import logging

from datetime import timedelta
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist

from elf.utils import TensorMetadata

logger = logging.getLogger(__name__)

__all__ = [
	"Placement",
	"PipelineTopology",
	"pipeline_size",
	"pp_members",
	"dp_members",
	"p2p_pairs",
	"send_models",
	"recv_models",
	"broadcast_models",
]


CTAS_PER_COMMUNICATOR = os.getenv("ELF_CTAS_PER_COMM", None)


class Placement(List[int]):
	"""
	Device placement pattern for pipeline stages.

	A placement is a list where each element specifies which device (rank) hosts that pipeline
	stage. Different schedulers require different placement patterns:

	- Linear: ``[0, 1, 2, 3]`` - each stage on a different device
	- Interleaved: ``[0, 1, 2, 3, 0, 1, 2, 3]`` - multiple stages per device
	- V-schedule: ``[0, 1, 2, 3, 3, 2, 1, 0]`` - bidirectional pattern

	The ``ranks`` attribute lists every device used, in increasing order.

	Example: ::

		placement = Placement([0, 1, 2, 3])  # manual placement
		placement = Placement.default("zbh2", pp=4)  # [0, 1, 2, 3]
		placement = Placement.default("zbv", pp=4)  # [0, 1, 2, 3, 3, 2, 1, 0]

	:param placement: device hosting each stage
	:type placement: List[int]
	"""

	def __init__(self, placement: List[int]):
		super().__init__(placement)
		assert not dist.is_initialized() or max(self) < dist.get_world_size(), (
			"Placement is out of bounds"
		)
		self.ranks = sorted(list(set(placement)))  # all unique processors, in increasing order

	def get_ids(self, rank: int) -> List[int]:
		"""
		Get the stage ids assigned to a specific rank.

		Example: ::

			placement = Placement([0, 1, 1, 2])
			placement.get_ids(1)  # [1, 2]

		:param rank: device rank to query
		:type rank: int

		:return: stage indices hosted by that rank
		:rtype: List[int]
		"""
		return [i for i in range(len(self)) if self[i] == rank]

	@staticmethod
	def default(scheduler, pp):
		"""
		Get the default placement pattern for a scheduler.

		Different pipeline schedulers have different placement requirements:

		- 1f1b, gpipe, zbh1, zbh2: linear placement ``[0, 1, 2, ...]``
		- megatron: interleaved placement ``[0, 1, 2, ..., 0, 1, 2, ...]``
		- hanayo, zbv: V-schedule ``[0, 1, 2, ..., 2, 1, 0]``

		Example: ::

			Placement.default("1f1b", pp=4)  # [0, 1, 2, 3]
			Placement.default("megatron", pp=4)  # [0, 1, 2, 3, 0, 1, 2, 3]
			Placement.default("zbv", pp=4)  # [0, 1, 2, 3, 3, 2, 1, 0]

		:param scheduler: name of the scheduler, as registered in :py:data:`elf.registry.SCHEDULERS`
		:type scheduler: str
		:param pp: number of pipeline parallel devices
		:type pp: int

		:return: the placement that scheduler expects
		:rtype: Placement
		"""
		if callable(scheduler):
			logger.warning(
				"Placement.default() expects a scheduler name, not the scheduler object itself. Using the default placement."
			)
			scheduler = ""

		if scheduler in ("hanayo", "zbv", "vmin", "vhalf"):
			return Placement([i for i in range(pp)] + list(reversed([i for i in range(pp)])))
		elif scheduler == "megatron":
			return Placement([i for i in range(pp)] * 2)
		else:
			return Placement([i for i in range(pp)])

	def head(self):
		"""
		Rank hosting the first stage of the pipeline.

		:rtype: int
		"""
		return self[0]

	def tail(self):
		"""
		Rank hosting the last stage of the pipeline.

		:rtype: int
		"""
		return self[-1]

	def add_offset(self, offset: int):
		"""
		Shift every rank, to address the pipeline of another data-parallel replica.

		:param offset: value added to every rank
		:type offset: int

		:return: the shifted placement
		:rtype: Placement
		"""
		return Placement([i + offset for i in self])

	def is_tail(self, rank: int):
		"""
		Whether a rank hosts the last stage of the pipeline.

		:param rank: device rank to query
		:type rank: int

		:rtype: bool
		"""
		return rank == self[-1]

	def is_head(self, rank: int):
		"""
		Whether a rank hosts the first stage of the pipeline.

		:param rank: device rank to query
		:type rank: int

		:rtype: bool
		"""
		return rank == self[0]

	def __add__(self, other):
		return Placement(list(self) + list(other))

	def __mul__(self, other):
		return Placement(list(self) * other)


# -- Checkers for placement validity on schedulers


def is_single_wave(placement: Placement, *_):
	return len(set(placement)) == len(placement)  # if no duplicates, it's singlewave


def is_cyclic(placement: Placement, *_):
	if len(placement) % len(placement.ranks) != 0:
		return False

	return placement == list(placement.ranks) * (len(placement) // len(placement.ranks))


def is_mirror(placement: Placement, *_):
	if len(placement) % len(placement.ranks) != 0:
		return False

	mirror_wave = list(placement.ranks) + list(reversed(placement.ranks))
	nwaves = len(placement) // (len(mirror_wave))

	return placement == nwaves * mirror_wave


def not_(f):  # functionalization of not
	return lambda *args, **kwargs: not f(*args, **kwargs)


def and_(f, g):  # functionalization of and
	return lambda *args, **kwargs: f(*args, **kwargs) and g(*args, **kwargs)


def or_(f, g):  # functionalization of and
	return lambda *args, **kwargs: f(*args, **kwargs) or g(*args, **kwargs)


def true_(*args, **kwargs):
	return True


# --


def pipeline_size(world_size: int, pp: int | None = None, dp: int = 1) -> int:
	"""
	Number of processes used for pipeline parallelism.

	:param world_size: total number of processes
	:type world_size: int
	:param pp: requested pipeline-parallel size, or None to fill the world
	:type pp: int or None
	:param dp: data-parallel size
	:type dp: int

	:return: the pipeline-parallel size
	:rtype: int
	"""
	pp = pp or world_size // dp
	assert pp * dp == world_size, (
		f"Requested PP = {pp}, DP = {dp} need {pp * dp} processes, but {world_size} were spawned"
	)
	return pp


def pp_members(world_size: int, pp: int, dp: int) -> list[list[int]]:
	"""
	Ranks of every pipeline-parallel group, one group per data-parallel replica.

	:return: ``members[replica]`` are the ranks running the pipeline of that replica
	:rtype: list[list[int]]
	"""
	return [[r for r in range(world_size) if r // pp == replica] for replica in range(dp)]


def dp_members(placement: Placement, pp: int, dp: int) -> list[list[int]]:
	"""
	Ranks of every data-parallel group, one group per stage.

	:param placement: the ORIGINAL placement (of the first replica)
	:type placement: Placement
	:return: ``members[position]`` are the ranks holding the same stages in different replicas
	:rtype: list[list[int]]
	"""
	return [[placement[stage] + pp * d for d in range(dp)] for stage in placement]


def p2p_pairs(signatures) -> list[tuple[int, int]]:
	"""
	Every pair of blocks that exchanges tensors, as ``(a, b)`` with ``a < b``.

	Both directions of the dataflow give the same pair: an activation sent from ``a`` to ``b`` in the
	forward pass comes back as a gradient in the backward pass. Blocks are numbered independently of
	the ranks running them, so every process computes the same list from the same signatures.

	:param signatures: dataflow signature of every block of the pipeline
	:type signatures: list[Signature]

	:return: the pairs, sorted, without duplicates
	:rtype: list[tuple[int, int]]
	"""
	pairs = set()
	for block_id, signature in enumerate(signatures):
		peers = signature.get_all_sources() + signature.get_all_targets()
		for peer in peers:
			if peer is not None and peer != block_id:
				pairs.add((min(block_id, peer), max(block_id, peer)))

	return sorted(pairs)


class PipelineTopology:
	"""
	Mapping between ranks, pipeline stages and process groups.

	Creating the topology creates the pipeline-parallel and data-parallel groups. The point-to-point
	groups need the dataflow signatures, which are only known once the model has been partitioned,
	so they are created later by :func:`build_p2p_groups`.

	:param placement: device placement of each stage, or "auto" to use the scheduler's default
	:type placement: Placement or list[int] or str
	:param pp: number of processes used for pipeline parallelism (default: fill the world)
	:type pp: int or None
	:param dp: number of data-parallel replicas (default: 1)
	:type dp: int
	:param scheduler: name of the scheduler, only used to pick the default placement
	:type scheduler: str or None
	:param timeout: process group timeout, in seconds (default: the ``ELF_TIMEOUT`` variable)
	:type timeout: int or None
	"""

	def __init__(self, placement="auto", pp=None, dp=1, scheduler=None, timeout=None):
		self.world_size = dist.get_world_size()
		self.rank = dist.get_rank()

		self.pp = pipeline_size(self.world_size, pp, dp)
		self.dp = dp
		self.dp_rank = self.rank // self.pp  # which replica this process belongs to
		self.pp_rank = self.rank % self.pp  # position of this process inside its replica

		if placement == "auto":
			placement = Placement.default(scheduler, self.pp)  # expects a name, not a scheduler

		# On each replica the placement is shifted to the actual ranks of that replica
		self.stage_placement = Placement(placement)  # replica-independent, stage -> position
		self.placement = self.stage_placement.add_offset(self.dp_rank * self.pp)

		if timeout is None:
			timeout = os.getenv("ELF_TIMEOUT", None)
		self.timeout = timedelta(seconds=int(timeout)) if timeout is not None else None

		self.pp_group = None
		self.dp_groups = []
		self._p2p_groups = {}
		# Groups this process belongs to, in creation order: they are torn down in reverse
		self._owned_groups = []

		self._init_groups()

	@classmethod
	def local(cls, placement) -> "PipelineTopology":
		"""
		Topology of a pipeline running in a single process, without any process group.
		Used for unit testing.

		:param placement: device placement of each stage
		:type placement: Placement or list[int]
		"""
		topology = cls.__new__(cls)
		topology.world_size = 1
		topology.rank = 0
		topology.pp = 1
		topology.dp = 1
		topology.dp_rank = 0
		topology.pp_rank = 0
		topology.stage_placement = Placement(placement)
		topology.placement = topology.stage_placement
		topology.timeout = None
		topology.pp_group = None
		topology.dp_groups = []
		topology._p2p_groups = {}
		topology._owned_groups = []
		return topology

	def _init_groups(self):
		"""
		Create the pipeline-parallel and data-parallel groups.

		Every process enters every ``new_group`` call, in the same order, as the API requires; each
		one only keeps the groups it belongs to.
		"""
		device = (
			torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else None
		)
		for replica, members in enumerate(pp_members(self.world_size, self.pp, self.dp)):
			if self.rank == members[0]:
				logger.debug(f"Creating PP group with members {members}")
			group = dist.new_group(members, device_id=device, timeout=self.timeout)
			if replica == self.dp_rank:
				self.pp_group = group
				self._owned_groups.append(group)

		if self.dp > 1:
			for members in dp_members(self.stage_placement, self.pp, self.dp):
				if self.rank == members[0]:
					logger.debug(f"Creating DP group with members {members}")
				group = dist.new_group(members, device_id=device, timeout=self.timeout)
				self.dp_groups.append(group)
				if self.rank in members:
					self._owned_groups.append(group)

	def build_p2p_groups(self, signatures):
		"""
		Create the process groups used for point-to-point communications.

		By default, torch distributed uses one cuda stream per pair of ranks. Bidirectional
		communications are therefore sequentialized on that stream, even if they are independent. We
		create separate groups for every type of communication (recv / send * fwd / bwd).

		:param signatures: dataflow signature of every block of the pipeline
		:type signatures: list[Signature]
		"""
		device = (
			torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else None
		)

		pairs = p2p_pairs(signatures)
		single_communicator = os.getenv("ELF_SINGLE_P2P_COMMUNICATOR", False)

		groups = {}
		for replica in range(self.dp):
			offset = replica * self.pp
			for pair in pairs:
				ranks = [self.stage_placement[block] + offset for block in pair]
				if ranks[0] == ranks[1]:  # both blocks on the same process, no communication needed
					continue

				if self.rank == ranks[0]:
					logger.debug(f"{self.rank} - Ranks {ranks} creating new PGs for P2Ps of {pair}")

				if not single_communicator:
					options = dist.ProcessGroupNCCL.Options()
					if CTAS_PER_COMMUNICATOR is not None:
						options.config.max_ctas = int(CTAS_PER_COMMUNICATOR)

					group_fwd = dist.new_group(ranks, device_id=device, pg_options=options)
					group_bwd = dist.new_group(ranks, device_id=device, pg_options=options)
				else:
					if dist.get_rank() == 0:
						logger.info("Using a single communicator for all p2ps. Performance may be degraded.")
					group_fwd = None
					group_bwd = None

				if replica == self.dp_rank:
					groups[pair] = (group_fwd, group_bwd)
					if self.rank in ranks:
						self._owned_groups += [group_fwd, group_bwd]

		# An activation travels forward and its gradient backward, on two separate groups
		for block_id in self.stage_ids():
			signature = signatures[block_id]
			recv_pgs = {"fwd": {}, "bwd": {}}
			send_pgs = {"fwd": {}, "bwd": {}}

			for source in signature.get_all_sources():
				pair = (min(block_id, source), max(block_id, source)) if source is not None else None
				if pair in groups:
					recv_pgs["fwd"][source], send_pgs["bwd"][source] = groups[pair]

			for target in signature.get_all_targets():
				pair = (min(block_id, target), max(block_id, target)) if target is not None else None
				if pair in groups:
					send_pgs["fwd"][target], recv_pgs["bwd"][target] = groups[pair]

			self._p2p_groups[block_id] = (recv_pgs, send_pgs)

	def p2p_groups(self, block_id):
		"""
		Get recv and send groups used for p2p communications of block `block_id`.

		:return: recv and send groups, each keyed by direction ("fwd" / "bwd") then by peer block
		:rtype: tuple[dict, dict]
		"""
		return self._p2p_groups.get(block_id, ({"fwd": {}, "bwd": {}}, {"fwd": {}, "bwd": {}}))

	def dp_group(self, block_id):
		"""
		Group over which the gradients of `block_id` are averaged, or None without data parallelism.

		:rtype: dist.ProcessGroup or None
		"""
		if self.dp <= 1:
			return None
		return self.dp_groups[block_id]  # use one group per stage

	def stage_ids(self, rank=None) -> list[int]:
		"""
		Stages hosted by a process (this one by default).

		:rtype: list[int]
		"""
		return self.placement.get_ids(self.rank if rank is None else rank)

	def is_first(self) -> bool:
		"""Whether this process owns the first stage of the pipeline."""
		return self.placement.is_head(self.rank)

	def is_last(self) -> bool:
		"""Whether this process owns the last stage of the pipeline."""
		return self.placement.is_tail(self.rank)

	def destroy(self):
		"""
		Destroy every process group this topology created.

		Destroying a group makes its members rendez-vous, so two processes sharing two groups have
		to destroy them in the same relative order or they deadlock, each waiting on the group the
		other has not reached yet. Groups are created in the same order on every process, so undoing
		that order -- which every process also agrees on -- is enough.
		"""
		for group in reversed(self._owned_groups):
			dist.destroy_process_group(group)

		self._owned_groups.clear()
		self._p2p_groups.clear()
		self.dp_groups.clear()
		self.pp_group = None

	def __repr__(self) -> str:
		return f"PipelineTopology(pp={self.pp}, dp={self.dp}, placement={list(self.placement)})"


def send_models(models, dst, group=None):
	"""
	Send a list of models using p2p comms
	Send the model structure (nn.Module object, parameters metadata, etc) via pickled CPU comm, and the actual tensors via GPU comm.

	:param models: list of models to send
	:type models: list[nn.Module]
	:param dst: destination rank
	:type dst: int
	:param group: communication group
	:type group: dist.ProcessGroup | None
	"""

	# Helper to split a qualified name like "a.b.weight" -> ("a.b", "weight")
	def _split_qualified_name(qname: str):
		if "." in qname:
			path, leaf = qname.rsplit(".", 1)
			return path, leaf
		return "", qname

	assert dist.is_initialized(), "torch.distributed must be initialized before calling send_models"

	# 1) Build tensor-free payloads and gather tensors to transfer via GPU p2p
	payloads = []
	per_model_params = []
	per_model_buffers = []
	saved_params = []
	saved_buffers = []

	for model in models:
		# Collect parameter names and tensors
		param_names = []
		param_requires = []
		param_tensors = []
		model_saved_params = []

		for name, p in model.named_parameters(recurse=True):
			param_names.append(name)
			param_requires.append(bool(p.requires_grad))
			pt = p.data
			if not pt.is_cuda:
				pt = pt.to(device="cuda")
			else:
				pt = pt.to(device=torch.cuda.current_device())
			param_tensors.append(pt.contiguous())

			module_path, leaf = _split_qualified_name(name)
			submod = model.get_submodule(module_path) if module_path != "" else model
			# Save original parameter object so we can restore it after pickling
			model_saved_params.append((submod, leaf, p))

			# During pickling (triggered by send_object_list), some module types
			# such as torch.fx.GraphModule may perform symbolic tracing and expect
			# parameters to be non-None. However, we do not want to serialize the
			# full GPU tensors on the CPU channel.
			#
			# To avoid GPU->CPU transfers while still satisfying these expectations,
			# we temporarily replace the parameter with a meta placeholder; the real GPU tensor
			# is sent separately below via p2p.
			dummy_param = nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=False)
			submod._parameters[leaf] = dummy_param  # type: ignore

		# Collect buffer names and tensors
		buffer_names = []
		buffer_tensors = []
		model_saved_buffers = []

		for name, b in model.named_buffers(recurse=True):
			buffer_names.append(name)
			bt = b
			if not bt.is_cuda:
				bt = bt.to(device="cuda")
			else:
				bt = bt.to(device=torch.cuda.current_device())
			buffer_tensors.append(bt.contiguous())

			module_path, leaf = _split_qualified_name(name)
			submod = model.get_submodule(module_path) if module_path != "" else model
			# Save original buffer so we can restore it after pickling
			model_saved_buffers.append((submod, leaf, b))
			# Same rationale as for parameters above: avoid serializing large
			# GPU tensors while keeping non-None buffers for modules that rely
			# on them during pickling.
			dummy_buffer = nn.Parameter(torch.empty_like(b, device="meta"), requires_grad=False)
			submod._buffers[leaf] = dummy_buffer

		payloads.append((model, param_names, param_requires, buffer_names))
		per_model_params.append(param_tensors)
		per_model_buffers.append(buffer_tensors)
		saved_params.append(model_saved_params)
		saved_buffers.append(model_saved_buffers)

	# 2) Send tensor-free module payloads via CPU object communication
	dist.send_object_list(payloads, dst=dst, group=group)

	# 3) Restore original params/buffers in local models
	for model_saved_params in saved_params:
		for submod, leaf, p in model_saved_params:
			submod._parameters[leaf] = p
	for model_saved_buffers in saved_buffers:
		for submod, leaf, b in model_saved_buffers:
			submod._buffers[leaf] = b

	# 4) Send tensor metadata and data via GPU p2p
	for model_params, model_buffers in zip(per_model_params, per_model_buffers):
		for t in model_params:
			meta = TensorMetadata(t).to_tensor()
			dist.send(meta, dst=dst, group=group)
			dist.send(t, dst=dst, group=group)
		for t in model_buffers:
			meta = TensorMetadata(t).to_tensor()
			dist.send(meta, dst=dst, group=group)
			dist.send(t, dst=dst, group=group)


def recv_models(models, src, group=None):
	"""
	Receive a list of models using p2p comms. See :func:`send_models` for more details.

	:param models: list object with correct size, that will be populated with the received models
	:type models: list[Any]
	:param src: source rank
	:type src: int
	:param group: communication group
	:type group: dist.ProcessGroup | None
	"""
	assert dist.is_initialized(), "torch.distributed must be initialized before calling recv_models"

	# Receive tensor-free module payloads
	obj_list = [None for _ in range(len(models))]
	dist.recv_object_list(obj_list, src=src, group=group)

	# Receive tensors via GPU and attach to the received modules
	for i, payload in enumerate(obj_list):
		module, param_names, param_requires, buffer_names = payload

		# Parameters
		for name, req in zip(param_names, param_requires):
			meta_tensor = torch.empty(TensorMetadata.MAX_SIZE, device=torch.cuda.current_device())
			dist.recv(meta_tensor, src=src, group=group)
			meta = TensorMetadata.from_tensor(meta_tensor)
			tensor = torch.empty(tuple(meta.shape), dtype=meta.dtype, device=torch.cuda.current_device())
			dist.recv(tensor, src=src, group=group)

			if "." in name:
				module_path, leaf = name.rsplit(".", 1)
				submod = module.get_submodule(module_path)
			else:
				submod, leaf = module, name
			param_obj = nn.Parameter(tensor, requires_grad=bool(req))
			submod.register_parameter(leaf, param_obj)

		# Buffers
		for name in buffer_names:
			meta_tensor = torch.empty(TensorMetadata.MAX_SIZE, device=torch.cuda.current_device())
			dist.recv(meta_tensor, src=src, group=group)
			meta = TensorMetadata.from_tensor(meta_tensor)
			tensor = torch.empty(tuple(meta.shape), dtype=meta.dtype, device=torch.cuda.current_device())
			dist.recv(tensor, src=src, group=group)

			if "." in name:
				module_path, leaf = name.rsplit(".", 1)
				submod = module.get_submodule(module_path)
			else:
				submod, leaf = module, name
			submod.register_buffer(leaf, tensor)

		module.cuda()
		models[i] = module


def broadcast_models(models, src, group=None):
	"""
	Broadcast a list of models using p2p comms.

	.. warning::
		Efficient broadcasting the same way as :func:`send_models` and :func:`recv_models` is not supported yet. This function may take a lot of time if sending GPU models.
	"""
	dist.broadcast_object_list(models, src=src, group=group)
	for model in models:
		model.cuda()
