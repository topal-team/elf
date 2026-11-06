"""
Pipeline API and orchestration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List
from dataclasses import asdict, dataclass, field, fields

import torch
import torch.distributed as dist

from elf.execution import PipelineBlock, Engine
from elf.scheduling import (
	schedule_to_str,
	check_schedule_validity,
	add_comms,
	mark_replayable_forwards,
)
from elf.profiling import profile_pipeline
from elf.scheduling.ilp import stagerematify, AutoScheduler
from elf.utils import TensorMetadata
from elf.registry import (
	resolve,
	Tracer,
	Partitioner,
	Scheduler,
	CommScheduler,
	TRACERS,
	PARTITIONERS,
	SCHEDULERS,
	COMM_SCHEDULERS,
)
from elf.partitioners import partition
from elf.signature import Signature, sequential_signatures
from elf.topology import Placement, PipelineTopology, send_models, recv_models, broadcast_models

import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
	r"""
	Configuration for Pipeline initialization.
	Registry arguments (tracer, partitioner, scheduler & comm_scheduler) can take any value listed in the :py:mod:`elf.registry`.
	Call ``elf.registry.[REGISTRY].available()`` to list the available values.

	:param tracer: Tracer to use for model graph extraction. When using default, tries all available tracers one after the other. (default: "default")
	:type tracer: Tracer or str
	:param partitioner: Partitioning strategy or False to skip partitioning (default: "balanced")
	:type partitioner: Partitioner or str
	:param scheduler: Static scheduling algorithm (default: "1f1b").\
		Can be "auto" to solve the ILP for all base schedulers and use the best objective value.
	:type scheduler: Scheduler or str
	:param comm_scheduler: Communication scheduler to use for reordering communications. (default: "timed-overlap")
	:param profiling: Measure every stage at setup (:func:`elf.profiling.profile_pipeline`) and hand\
		the result to the scheduler and to the communication scheduler. Requires ``sample``. (default: True)
	:type profiling: bool
	:param placement: Device placement strategy as a list that maps stage id -> processor id (default: "auto")
	:type placement: Placement or str
	:param pp: Number of pipeline parallel processes. If None, computed as world_size // dp
	:type pp: int or None
	:param dp: Number of data parallel processes (default: 1)
	:type dp: int
	:param worker: Rank that profiles and partitions the model (default: 0)
	:type worker: int
	:param memory_budgets: GPU memory budget in MB for ILP-based rematerialization.
	When set, stages are profiled at partition time and an ILP is solved on the first training step to decide which operations to recompute. (default: None)
	Can be either one budget per rank or a single budget used for every rank
	:type memory_budgets: int, List[int] or None
	:param signatures: Dataflow signatures describing the communication pattern between pipeline stages.\
	Only needed when ``partitioner=False``, i.e. your model is already partitioned. \
		Use :func:`elf.signature.sequential_signatures` for simple sequential models. \
		If not provided, a sequential pattern is assumed.
	:type signatures: List[Signature] or None
	"""

	# Registry fields (need to be resolved)
	tracer: Tracer | str = field(default="default", metadata={"registry": TRACERS})
	partitioner: Partitioner | str = field(default="balanced", metadata={"registry": PARTITIONERS})
	scheduler: Scheduler | str = field(default="auto", metadata={"registry": SCHEDULERS})
	comm_scheduler: CommScheduler | str = field(
		default="timed-overlap", metadata={"registry": COMM_SCHEDULERS}
	)
	# Regular fields
	profiling: bool = True
	placement: Placement | str = "auto"
	pp: int | None = None
	dp: int = 1
	worker: int = 0
	memory_budgets: int | List[int] | None = None
	signatures: List[int, Signature] | None = None  # only needed when partitioner = False

	def resolve(self):
		resolved = {}
		for f in fields(self):
			val = getattr(self, f.name)
			registry = f.metadata.get("registry")
			if registry is not None and val is not False:
				resolved[f.name] = resolve(val, registry)
			else:
				resolved[f.name] = val

		return PipelineConfig(**resolved)

	def to_kwargs(self) -> dict[str, Any]:
		"""Convert config to kwargs dict, excluding None values."""
		return {k: v for k, v in asdict(self).items() if v is not None}


class Pipeline:
	"""
	Main API for pipeline parallelism with automatic model partitioning and scheduling.

	Wraps a model and handles tracing, partitioning across devices, distributed communication
	setup, and execution of pipeline schedules. Supports various scheduling algorithms (1F1B,
	GPipe, ZBH, etc.) and partitioning strategies. Can be combined with data parallelism.
	"""

	def __init__(
		self,
		model,
		sample=None,
		*,
		config: PipelineConfig | None = None,
		# Configuration parameters (used when config is None)
		**kwargs,
	):
		"""
		Initialize a Pipeline with model parallelism configuration.

		**Simple usage (no config object needed):**

		.. code-block:: python

			pipeline = Pipeline(model, sample, scheduler="1f1b", dp=2)

		**Usage with PipelineConfig (recommended for complex setups):**

		.. code-block:: python

			config = PipelineConfig(scheduler="zbh2", dp=2, pp=4)
			pipeline = Pipeline(model, sample, config=config)

		When `config` is provided, it takes precedence and individual parameters are ignored.

		:param model: the entire model to pipeline, or this rank's portion of a pre-partitioned model
		:type model: nn.Module
		:param sample: sample inputs used for profiling. Not needed when using pre-partitioned model. The batch size should be the size of an individual micro-batch.
		:type sample: torch.Tensor or List[torch.Tensor]
		:param config: PipelineConfig object for centralized configuration. When provided, individual parameters are ignored. If not provided, individual fields can be filled via kwargs. All fields that are not provided will use default values. See :class:`elf.pipeline.PipelineConfig` for more details.
		:type config: PipelineConfig or None
		"""
		if not dist.is_initialized():
			logger.warning(
				"Trying to create a pipeline but no multi-gpu distributed setup has been found."
			)
		# Create config from either provided config or individual parameters
		if config and kwargs and dist.get_rank() == 0:
			logger.warning(
				"Both PipelineConfig and named arguments provided. Named arguments will be ignored. Write them to the PipelineConfig object instead."
			)

		self.cfg = config or PipelineConfig(**kwargs)
		self.cfg = self.cfg.resolve()

		# Distributed setup: ranks, placement and process groups
		self.topology = PipelineTopology(
			placement=self.cfg.placement,
			pp=self.cfg.pp,
			dp=self.cfg.dp,
			scheduler=self._default_placement_hint(),
		)

		# Components are already resolved in the config
		self.scheduler = self.cfg.scheduler
		self.tracer = self.cfg.tracer
		self.partitioner = self.cfg.partitioner  # False if no partitioning is desired

		# Partition model
		parts, signatures = self._partition_model(model, sample, self.cfg.signatures)
		self.signatures = signatures

		# Create execution engine
		self.blocks = self._create_pipeline(parts, signatures)
		self.engine = Engine(self.blocks)

		# Used to avoid re-generating schedule every time
		self.schedule = []
		self.last_nmb = 0

		# Measure the stages once, for everything downstream that reasons about time or memory
		self.profiles = self._profile_stages(sample)

		# ILP remat: solve lazily at first step(), once the number of micro-batches is known
		self._remat_scheduler = None
		self._remat_nmb = None
		if self.cfg.memory_budgets is not None:
			assert self.profiles is not None, (
				"memory_budget needs the profiling statistics: pass a sample and leave profiling on"
			)
			if type(self.cfg.memory_budgets) is int:
				self.cfg.memory_budgets = [self.cfg.memory_budgets for _ in range(self.topology.pp)]

			if self.scheduler is SCHEDULERS.get("auto"):
				# Special case: auto scheduler solves StageRemat for all base schedulers and uses the best objective value
				self.scheduler = AutoScheduler(self.cfg.memory_budgets, self.cfg.worker)
			else:
				self.scheduler = stagerematify(self.scheduler, self.cfg.memory_budgets, self.cfg.worker)

	def _default_placement_hint(self):
		"""
		Name of the scheduler, used by the topology to pick the default placement.

		:return: the registry key of the scheduler, or None for a scheduler that is not registered
		:rtype: str or None
		"""
		if self.cfg.placement != "auto":
			return None
		try:
			return SCHEDULERS.get_key(self.cfg.scheduler)
		except ValueError:
			return None  # a custom scheduler: fall back to the default placement

	@property
	def placement(self):
		"""Device placement of every stage, shifted to the ranks of this data-parallel replica."""
		return self.topology.placement

	@property
	def pp(self):
		"""Number of processes used for pipeline parallelism."""
		return self.topology.pp

	@property
	def dp(self):
		"""Number of data-parallel replicas."""
		return self.topology.dp

	@property
	def pp_group(self):
		"""Process group of this replica's pipeline."""
		return self.topology.pp_group

	@property
	def dp_groups(self):
		"""Process groups over which gradients are averaged, one per position in the pipeline."""
		return self.topology.dp_groups

	def _clear_blocks(self):
		for block in self.blocks:
			block._wait_for_send_ops()
			block.sync_all_reduce()
			for var in block.input_variables:
				var.clear()
			for var in block.output_variables:
				for dst in var:
					dst.clear()
			for module in block._layers_dw:
				module.clear()

	def step(self, batch, target, loss_fn, split_size=0, profile=False, scheduler=None):
		"""
		Execute a training step on a batch of data using the pipeline schedule.

		:param batch: input data
		:type batch: torch.Tensor or List[torch.Tensor]
		:param target: targets. Can be None for inference.
		:type target: torch.Tensor or None
		:param loss_fn: loss function to be used. We recommend using torch's built-in loss functions, but you can pass any function that matches the signature. Be careful, the loss is computed on each micro-batch, then averaged over the batch dimension. Depending on the loss function, this may not be equivalent to computing the loss on the full batch.
		:type loss_fn: Function (Tensor,Tensor) -> Tensor
		:param split_size: either one size for equal micro batches (last one may be smaller if the batch size is not divisible by the split size), or a list of possibly different micro batch sizes. In that case the sum of the sizes must be equal to the batch size. Default value is (batch_size // number of gpus)
		:type split_size: int or List[int]
		:param profile: Whether to activate nvidia profiling or not. If True, NVTX ranges will be generated for each operation
		:type profile: boolean
		:param scheduler: scheduler to use for the step. If None, the one passed to the constructor is used.
		:type scheduler: Scheduler or None

		:return: result of the forward pass and loss value if the last block of the pipeline is managed by this process
		:rtype: (List[Tensor], Tensor) or (None, None)

		.. warning::
			The result is automatically offloaded to CPU. Since merging it back would cause a sync point, it is currently returned as a list of tensors, one for each micro-batch. Avoid merging it back to a single tensor between iterations if you want to avoid performance issues.
		"""
		if batch is not None:
			# We expect a list of arguments, not a single tensor
			if isinstance(batch, torch.Tensor):
				batch = [batch]

		mb_sizes = self._get_mb_sizes(split_size, batch)
		n_micro_batches = len(mb_sizes)

		# we consider that a schedule is fixed for a given algorithm and number of micro-batches
		regenerate_schedule = scheduler is not None or n_micro_batches != self.last_nmb
		# Resolve scheduler to use
		if scheduler is None:
			scheduler = self.scheduler
		else:
			scheduler = resolve(scheduler, SCHEDULERS)

		if regenerate_schedule:
			# We have to recompute the schedule
			self._generate_schedule(n_micro_batches, scheduler, self.cfg.comm_scheduler)

			self.last_nmb = n_micro_batches

		self._clear_blocks()  # clear before to make sure iteration starts clean

		# Execute the schedule
		result, losses, stats, detailed_stats = self.engine.train_step(
			batch, target, loss_fn, self.schedule, mb_sizes, profile
		)

		self._clear_blocks()  # clear after to avoid keeping unnecessary tensors in memory

		self.stats = stats
		self.detailed_stats = detailed_stats

		# Merge back the micro-batches outputs/losses into one batch
		if any(r is not None for r in result):
			with torch.no_grad():
				# Careful! if the result was offloaded to CPU, then this creates a sync point that slows down the execution
				# result = torch.cat(result, dim=0)
				# This causes a stream synchronization ; maybe it can be avoided
				# From looking at the profiler, it does not incur important overhead, but need to check
				losses = torch.stack(losses).sum()

			if self.dp > 1:
				dist.all_reduce(losses, group=self.blocks[-1].dp_group, op=dist.ReduceOp.AVG)

			return result, losses
		else:
			return None, None

	def __call__(self, *args, **kwargs):
		return self.step(*args, **kwargs)

	def parameters(self):
		"""
		Return an iterator over all parameters in this process's pipeline blocks.

		:return: Parameter iterator
		:rtype: Iterator[torch.nn.Parameter]
		"""
		for block in self.blocks:
			for param in block.model.parameters():
				yield param

	def named_parameters(self):
		"""
		Return an iterator over named parameters in this process's pipeline blocks.

		:return: Iterator over (name, parameter) tuples
		:rtype: Iterator[Tuple[str, torch.nn.Parameter]]
		"""
		for block in self.blocks:
			for name, param in block.model.named_parameters():
				yield name, param

	def zero_grad(self, set_to_none=True):
		"""
		Zero out gradients of all parameters in this process's pipeline blocks.

		:param set_to_none: If True, sets gradients to None instead of zero (default: True)
		:type set_to_none: bool
		"""
		for block in self.blocks:
			block.model.zero_grad(set_to_none=set_to_none)

	def clear(self):
		"""
		Synchronize, then destroy all DP and PP process groups.
		"""
		torch.cuda.synchronize()
		dist.barrier()
		self.topology.destroy()

	def checkpoint(self, epoch="init", dir_path="./"):
		"""
		Save model state to disk. Creates one file per rank named ``dp{dp}_pp{pp}.pt``.

		:param epoch: Epoch identifier, creates subdirectory ``{dir_path}/{epoch}/`` (default: "init")
		:type epoch: str
		:param dir_path: Base directory for checkpoints (default: "./")
		:type dir_path: str
		"""
		rank = dist.get_rank()
		dp, pp = self.topology.dp_rank, self.topology.pp_rank
		rank_state_dict = {"rank": rank, "pp": pp, "dp": dp, "epoch": epoch}

		for block in self.blocks:
			rank_state_dict[f"state_dict_{block.id}"] = {
				k: v.data for k, v in block.model.state_dict().items()
			}

		dir_path = Path(dir_path) / str(epoch)
		dir_path.mkdir(parents=True, exist_ok=True)
		torch.save(rank_state_dict, f"{dir_path}/dp{dp}_pp{pp}.pt")

	def save(self, path, worker=0):
		"""
		Gather parameters from all PP ranks and save to a single file.

		.. warning::
			Should not be used with `partitioner=False`.

		:param path: File path for saved model state
		:type path: str
		:param worker: Rank that gathers and writes the file (default: 0)
		:type worker: int
		"""
		rank = dist.get_rank()
		pp_group = self.blocks[0].pp_group
		if rank == worker:
			full_state = {}
			n_devices = max(self.placement) + 1
			for d in range(n_devices):
				if d == worker:
					for p_name, p in self.named_parameters():
						full_state[p_name] = p.cpu().detach()
				else:
					n_parts = len(self.placement.get_ids(d))
					parts = [None for _ in range(n_parts)]

					recv_models(parts, src=d, group=pp_group)

					for part in parts:
						for p_name, p in part.named_parameters():
							full_state[p_name] = p.to("cpu", non_blocking=True).detach()

			torch.cuda.synchronize()
			torch.save(full_state, path)
			logger.info(f"Saved model to {path}")

		else:
			if worker in dist.get_process_group_ranks(pp_group):
				parts = [block.model for block in self.blocks]
				send_models(parts, dst=worker, group=pp_group)

	def gather_parameters(self, dst=0):
		"""
		Gather parameters from all PP ranks to destination rank.

		:param dst: Rank that receives all parameters (default: 0)
		:type dst: int
		:return: State dict on dst rank, empty dict on others
		:rtype: dict[str, torch.Tensor]
		"""
		rank = dist.get_rank()
		pp_group = self.blocks[0].pp_group

		# Build local state dict for each rank
		local_params = {}
		for block in self.blocks:
			for name, param in block.model.named_parameters():
				if name not in local_params:
					local_params[name] = param.data
				else:
					logger.warning(
						f"Duplicate parameter {name} found in block {block.id}, using first occurrence"
					)

		# Create and gather metadata on cpu-cpu communication
		metadata = {name: TensorMetadata(tensor) for name, tensor in local_params.items()}
		all_metadata = [None] * dist.get_world_size() if rank == dst else None
		dist.gather_object(metadata, all_metadata, dst=dst, group=pp_group)

		if rank == dst:
			# Send actual parameters via gpu-gpu communication
			state_dict = local_params.copy()  # prefill with local parameters
			for peer, peer_metadata in enumerate(all_metadata):
				if peer != rank:
					for name, meta in peer_metadata.items():
						buffer = meta.get_buffer(1).squeeze(0)
						dist.recv(buffer, src=peer, group=pp_group)
						state_dict[name] = buffer
			return state_dict
		else:
			for tensor in local_params.values():
				dist.send(tensor, dst=dst, group=pp_group)
			return {}

	def is_first(self):
		"""Check if this process owns the first pipeline stage."""
		return self.topology.is_first()

	def is_last(self):
		"""Check if this process owns the last pipeline stage."""
		return self.topology.is_last()

	def _profile_stages(self, sample):
		"""
		Measure every stage of the pipeline, unless asked not to or unable to.

		Profiling replays each stage a few dozen times, so it needs a sample to replay it on; a
		pre-partitioned model built without one simply goes unprofiled, and the components that
		would have used the numbers fall back to their structural defaults.

		:param sample: the sample the pipeline was built with
		:return: one profile per stage, or None
		:rtype: List[StageProfile] or None
		"""
		if not self.cfg.profiling:
			return None
		if sample is None:
			logger.warning(
				"No sample was given, so the stages cannot be profiled. The scheduler and the"
				" communication scheduler will fall back to reasoning about the order of the"
				" operations alone."
			)
			return None
		return profile_pipeline(self.blocks, self.signatures, self.placement, sample, self.pp_group)

	def _get_mb_sizes(self, split_size, batch):
		"""
		Compute micro-batch sizes. If split_size is 0, defaults to batch_size // pp.

		:param split_size: Size per micro-batch (int) or list of sizes (must sum to batch_size)
		:type split_size: int or List[int]
		:param batch: Input batch
		:type batch: List[torch.Tensor]
		:return: List of micro-batch sizes
		:rtype: List[int]
		"""
		# Split size can be an int or list of ints ; make it always a list
		# Assuming all elements in the list have the same batch size
		batch_size = batch[0].size(0)
		if isinstance(split_size, int):
			if split_size > batch_size:
				logger.warning(f"Split size {split_size} is greater than batch size {batch_size}")
				split_size = batch_size
			if split_size == 0:
				split_size = batch_size // self.pp if self.pp <= batch_size else 1
			n_micro_batches = batch_size // split_size
			mb_sizes = [split_size for _ in range(n_micro_batches)]
			if batch_size % split_size != 0:
				mb_sizes.append(batch_size % split_size)
		else:
			assert sum(split_size) == batch_size, "Splits do not cover the entire batch"
			mb_sizes = split_size

		return mb_sizes

	def _generate_schedule(self, n_micro_batches, scheduler, comm_scheduler):
		"""
		Generate and optimize the execution schedule, then filter for current process.

		:param n_micro_batches: Number of micro-batches
		:type n_micro_batches: int
		:param scheduler: scheduler to use for the step
		:type scheduler: Scheduler
		:param comm_scheduler: communication scheduler to use for the step
		:type comm_scheduler: CommScheduler
		:param mb_sizes: List of micro-batch sizes
		:type mb_sizes: List[int]
		"""
		if not scheduler.check(self.placement, n_micro_batches):
			raise ValueError(
				f"{self.placement} is not valid with {n_micro_batches} microbatches for scheduler {scheduler}."
			)  # todo; write somewhere the constraints for each scheduler and redirect to this table from this message

		schedule = scheduler(self.placement, n_micro_batches, self.profiles)
		schedule = mark_replayable_forwards(schedule)
		schedule = add_comms(schedule, self.signatures)
		if comm_scheduler:
			schedule = comm_scheduler(schedule, self.profiles)

		check_schedule_validity(schedule)

		if dist.get_rank() == 0:
			logger.debug(f"Schedule:\n{schedule_to_str(schedule, True)}")

		# Remove all operations that are not ours
		ids = {block.id for block in self.blocks}
		self.schedule = list(filter(lambda op: op.block_id in ids, schedule))

	def _partition_model(self, model, sample, signatures):
		"""
		Partition the model or validate pre-partitioned model.

		:param model: Model to partition or list of pre-partitioned modules
		:type model: nn.Module or List[nn.Module]
		:param sample: Sample input for tracing (required if partitioner is set)
		:type sample: torch.Tensor or List[torch.Tensor] or None
		:param signatures: Dataflow signatures for pre-partitioned models
		:type signatures: List[Signature] or None
		:return: Model parts and signatures
		:rtype: Tuple[List[nn.Module], List[Signature]]
		"""
		if not self.partitioner:
			if isinstance(model, torch.nn.Module):
				parts = [model]
			else:
				parts = model

			if signatures is None:
				if dist.get_rank() == 0:
					logger.warning(
						"No signatures provided, assuming sequential partitioning. "
						"If this is not what you want, please explicitly provide signatures when creating the pipeline."
					)
				signatures = sequential_signatures(self.placement)
		else:
			assert sample is not None, "Sample is required for partitioning"
			try:
				parts, signatures = self._shared_partition(model, sample)
			except Exception as e:
				logger.error(
					"Error partitioning the model. This can be due to your model using features either not supported by torch.fx/torch.export, or by this library."
				)
				raise e

		return parts, signatures

	def _create_pipeline(self, parts, signatures):
		"""
		Create pipeline blocks for this process from partitioned model parts.

		:param parts: Model parts from partitioned model
		:type parts: List[nn.Module]
		:param signatures: Dataflow signatures for each block
		:type signatures: List[Signature]
		:return: Pipeline blocks for this process
		:rtype: List[PipelineBlock]
		"""
		# Initialize P2P process groups for all blocks this rank owns
		self.topology.build_p2p_groups(signatures)

		ids = self.topology.stage_ids()
		assert len(parts) == len(ids), (
			f"This process was given {len(parts)} parts but the placement assigns it {len(ids)} stages"
		)
		blocks = [
			PipelineBlock(part, block_id, signatures[block_id], self.topology)
			for part, block_id in zip(parts, ids)
		]

		dist.barrier(self.pp_group)  # init communicators for this process group with a collective

		return blocks

	def _shared_partition(self, model, sample):
		"""
		Partition model on worker rank and distribute to all processes.

		:param model: Model to partition
		:type model: nn.Module
		:param sample: Sample input for tracing
		:type sample: torch.Tensor or List[torch.Tensor]
		:return: Model parts for this process and signatures
		:rtype: Tuple[List[nn.Module], List[Signature]]
		"""
		rank = dist.get_rank()

		# Rank 'worker' profiles & partition the graph, then shares it to everyone
		n_blocks = self.placement.count(rank)
		blocks = [None for _ in range(n_blocks)]
		signatures = [None for _ in range(len(self.placement))]
		worker = self.cfg.worker
		if rank == worker:
			result = partition(
				model, len(self.placement), sample, partitioner=self.partitioner, tracer=self.tracer
			)
			parts = result.create_blocks()
			signatures = result.signatures

			for d in range(self.pp):
				blocks_on_d = [parts[i] for i in range(len(parts)) if self.placement[i] == d]
				if d == worker:
					blocks = blocks_on_d
					for block in blocks:
						block.cuda()
				else:
					send_models(blocks_on_d, dst=d, group=self.pp_group)
					dist.send_object_list(signatures, dst=d, group=self.pp_group)
					for block in blocks_on_d:
						block.cpu()

		elif rank < self.pp:
			# First pipeline share to their DP replicas
			recv_models(blocks, src=worker, group=self.pp_group)
			dist.recv_object_list(signatures, src=worker, group=self.pp_group)

		if self.dp > 1:
			r = rank % self.pp
			broadcast_models(blocks, src=r, group=self.dp_groups[r])
			dist.broadcast_object_list(signatures, src=r, group=self.dp_groups[r])

		logger.debug(f"Rank {rank} has {len(blocks)} block" + ("s" if len(blocks) > 1 else ""))

		return blocks, signatures
