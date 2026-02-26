"""
Pipeline API and orchestration
"""

import os

from dataclasses import asdict, dataclass
from typing import Any
from datetime import timedelta

import torch
import torch.distributed as dist

from .execution import PipelineBlock, Engine
from .scheduling import schedule_to_str, check_schedule_validity, reorder_communications
from .utils import TensorMetadata, send_models, recv_models, broadcast_models, Placement
from .zb_utils import LayerDW
from .registry import Tracer, Partitioner, Scheduler, TRACERS, PARTITIONERS, SCHEDULERS, resolve
from .partitioners import (
	partition_graph,
	signatures_from_sources_targets,
	get_sources_targets_sequential,
)

import logging

logger = logging.getLogger("pipeline")


@dataclass
class PipelineConfig:
	"""
	Configuration for Pipeline initialization.

	:param tracer: Tracer to use for model graph extraction. When using default, tries all available tracers one after the other. (default: "default")
	:type tracer: Tracer or str
	:param partitioner: Partitioning strategy or False to skip partitioning (default: "constrained")
	:type partitioner: Partitioner or str
	:param scheduler: Static scheduling algorithm (default: "1f1b")
	:type scheduler: Scheduler or str
	:param placement: Device placement strategy or list of ranks (default: "auto")
	:type placement: Placement or str
	:param pp: Number of pipeline parallel processes. If None, computed as world_size // dp
	:type pp: int or None
	:param dp: Number of data parallel processes (default: 1)
	:type dp: int
	:param worker: Rank that profiles and partitions the model (default: 0)
	:type worker: int
	"""

	tracer: Tracer | str = "default"
	partitioner: Partitioner | str = "constrained"
	scheduler: Scheduler | str = "1f1b"
	placement: Placement | str = "auto"
	pp: int | None = None
	dp: int = 1
	worker: int = 0

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
		placement="auto",
		partitioner="constrained",
		scheduler="1f1b",
		tracer="default",
		pp=None,
		dp=1,
		worker=0,
		sources=None,
		targets=None,
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
		:param sample: sample inputs used for profiling. Not needed when using pre-partitioned model.
		:type sample: torch.Tensor or List[torch.Tensor]
		:param config: PipelineConfig object for centralized configuration. When provided, individual parameters are ignored.
		:type config: PipelineConfig or None
		:param placement: list of device ranks. Block ``i`` of the pipeline will be placed on rank ``placement[i]``. Leave to default ("auto") for automatic placement.
		:type placement: List[int] or str
		:param partitioner: if your model is already partitioned, set to False. Otherwise set to the partition strategy you want to use (default = constrained), which will try to create balanced blocks according to their profiled execution time.
		:type partitioner: boolean or str
		:param scheduler: static scheduling algorithm to use. currently supported : GPipe ("afab"), PipeDream ("1f1b") (default), Hanayo ("hanayo"), ZBH1/ZBH2/ZBV ("zbh1", "zbh2", "zbv"), Full Remat ("full_remat"), Inference ("inference"). You can also define your own scheduler by registering it in the registry.
		:type scheduler: str or function(List[int], int, **kwargs) -> List[Operation]
		:param tracer: Tracer to use for model graph extraction. When using default, tries all available tracers one after the other. (default: "default")
		:type tracer: Tracer or str
		:param pp: Number of pipeline parallel processes. If None, computed as world_size // dp
		:type pp: int or None
		:param dp: number of data parallel processes to use.
		:type dp: int
		:param worker: rank of the process that will profile the model and partition it
		:type worker: int
		:param sources: For each stage of the entire model (not only this rank's portion), source block id for each input variable. Only needed when partitioner is False, i.e. your model is already partitioned. See :func:`partitioners.utils.get_sources_targets_sequential` for an example.
		:type sources: List[Dict[str, int]]
		:param targets: For each stage of the entire model (not only this rank's portion), target block ids for each output variable. Only needed when partitioner is False, i.e. your model is already partitioned. See :func:`partitioners.utils.get_sources_targets_sequential` for an example.
		:type targets: List[Dict[str, List[int]]]
		"""
		if not dist.is_initialized():
			logger.warning(
				"Trying to create a pipeline but no multi-gpu distributed setup has been found."
			)
		ws = dist.get_world_size()
		rank = dist.get_rank()

		# Create config from either provided config or individual parameters
		if config is not None:
			self.cfg = config
		else:
			self.cfg = PipelineConfig(
				tracer=tracer,
				partitioner=partitioner,
				scheduler=scheduler,
				placement=placement,
				pp=pp,
				dp=dp,
				worker=worker,
			)
		self.pp = self.cfg.pp or ws // self.cfg.dp  # Number of GPUs used for pipeline parallelism
		self.dp = self.cfg.dp  # Number of GPUs used for data parallelism
		assert self.pp * self.dp == ws, (
			f"Requested PP = {self.pp}, DP = {self.dp} need {self.pp * self.dp} processes, but {ws} were spawned"
		)

		# Distributed setups
		if self.cfg.placement == "auto":
			self.placement = Placement.default(self.cfg.scheduler, self.pp)
		else:
			self.placement = Placement(self.cfg.placement)

		self.dp_rank = rank // self.pp
		self.placement = self.placement.add_offset(
			self.dp_rank * self.pp
		)  # on each DP replica, placement is shifted to correspond to actual ranks of this replica

		self._init_process_groups()  # sets pp_group and dp_groups

		# Resolve components from registries
		self.scheduler = resolve(self.cfg.scheduler, SCHEDULERS)
		self.tracer = resolve(self.cfg.tracer, TRACERS)
		self.partitioner = (
			resolve(self.cfg.partitioner, PARTITIONERS) if self.cfg.partitioner else False
		)  # False if no partitioning is desired

		# Partition model
		parts, signatures = self._partition_model(model, sample, sources, targets)
		self.signatures = signatures

		# Create execution engine
		self.blocks = self._create_pipeline(parts, signatures)
		self.engine = Engine(self.blocks)

		# Used to avoid re-generating schedule every time
		self.schedule = []
		self.last_nmb = 0

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
			self._generate_schedule(n_micro_batches, scheduler)

			self.last_nmb = n_micro_batches

		for block in self.blocks:
			for module in block.model.modules():
				if isinstance(module, LayerDW):
					module.clear()

		# Execute the schedule
		result, losses, stats, detailed_stats = self.engine.train_step(
			batch, target, loss_fn, self.schedule, mb_sizes, profile
		)

		# First and last block have some remaining tensors
		for block in self.blocks:
			block._wait_for_send_ops()
			for var in block.input_variables:
				var.clear()
			for var in block.output_variables:
				for dst in var:
					dst.clear()

			for module in block.model.modules():
				if isinstance(module, LayerDW):
					module.clear()

		self.stats = stats
		self.detailed_stats = detailed_stats

		# Merge back the micro-batches outputs/losses into one batch
		if len(result) != 0:
			with torch.no_grad():
				# Careful! if the result was offloaded to CPU, then this creates a sync point that slows down the execution
				# result = torch.cat(result, dim=0)
				# This causes a stream synchronization ; maybe it can be avoided
				# From looking at the profiler, it does not incur important overhead, but need to check
				losses = torch.tensor(losses, device="cuda")
				losses = losses.sum() / sum(mb_sizes)

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
		for block in self.blocks:
			block._destroy_process_groups()

		logger.debug(
			f"Destroying PP group with members {dist.get_process_group_ranks(self.blocks[0].pp_group)}"
		)
		dist.destroy_process_group(self.blocks[0].pp_group)

	def checkpoint(self, epoch="init", dir_path="./"):
		"""
		Save model state to disk. Creates one file per rank named ``dp{dp}_pp{pp}.pt``.

		:param epoch: Epoch identifier, creates subdirectory ``{dir_path}/{epoch}/`` (default: "init")
		:type epoch: str
		:param dir_path: Base directory for checkpoints (default: "./")
		:type dir_path: str
		"""
		rank = dist.get_rank()
		dp = rank // self.pp
		pp = rank % self.pp
		rank_state_dict = {"rank": rank, "pp": pp, "dp": dp, "epoch": epoch}

		for block in self.blocks:
			rank_state_dict[f"state_dict_{block.id}"] = {
				k: v.data for k, v in block.model.state_dict().items()
			}

		dir_path = f"{dir_path}/{epoch}/"
		if not os.path.exists(dir_path):
			os.makedirs(dir_path)
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

					for part in enumerate(parts):
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
		return dist.get_rank() == self.placement[0]

	def is_last(self):
		"""Check if this process owns the last pipeline stage."""
		return dist.get_rank() == self.placement[-1]

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

	def _generate_schedule(self, n_micro_batches, scheduler):
		"""
		Generate and optimize the execution schedule, then filter for current process.

		:param n_micro_batches: Number of micro-batches
		:type n_micro_batches: int
		:param scheduler: scheduler to use for the step
		:type scheduler: Scheduler
		"""
		schedule = scheduler(self.placement, n_micro_batches, self.signatures)
		check_schedule_validity(schedule)

		schedule = reorder_communications(schedule, strategy="pipelined")

		if dist.get_rank() == 0:
			logger.debug(f"Schedule:\n{schedule_to_str(schedule)}")

		# Remove all operations that are not ours
		ids = list(
			map(lambda b: b.id, self.blocks)
		)  # funny python tips: a map in itself can be iterated only once ! never forget to create a list from it before anything else
		self.schedule = list(filter(lambda op: op.block_id in ids, schedule))

	def _partition_model(self, model, sample, sources, targets):
		"""
		Partition the model or validate pre-partitioned model.

		:param model: Model to partition or list of pre-partitioned modules
		:type model: nn.Module or List[nn.Module]
		:param sample: Sample input for tracing (required if partitioner is set)
		:type sample: torch.Tensor or List[torch.Tensor] or None
		:param sources: Source block IDs for pre-partitioned models
		:type sources: List[Dict[str, int]] or None
		:param targets: Target block IDs for pre-partitioned models
		:type targets: List[Dict[str, List[int]]] or None
		:return: Model parts and signatures
		:rtype: Tuple[List[nn.Module], List[Signature]]
		"""
		if not self.partitioner:
			# Model is already the right part, just make sure it's a list
			if isinstance(model, torch.nn.Module):
				parts = [model]
			else:
				parts = model

			if sources is None and targets is None:
				if dist.get_rank() == 0:
					logger.warning(
						"No sources and targets provided, assuming sequential partitioning. If this is not what you want, please explictly provide sources and targets when creating the pipeline."
					)
				sources, targets = get_sources_targets_sequential(self.placement)
			signatures = signatures_from_sources_targets(sources, targets)
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
		rank = dist.get_rank()

		ids = self.placement.get_ids(rank)

		# Initialize P2P process groups for all blocks this rank owns
		block_p2p_groups = self._init_p2p_process_groups(signatures)

		blocks = []
		for i in range(len(parts)):
			dp_group = self.dp_groups[ids[i] % self.pp] if self.dp > 1 else None
			recv_pgs, send_pgs = block_p2p_groups[ids[i]]
			new_block = PipelineBlock(
				parts[i],
				ids[i],
				self.placement,
				signatures[ids[i]],
				pp_group=self.pp_group,
				dp_group=dp_group,
				recv_pgs=recv_pgs,
				send_pgs=send_pgs,
			)
			blocks.append(new_block)

		dist.barrier(self.pp_group)  # init communicators for this process group with a collective

		return blocks

	def _init_process_groups(self):
		"""
		Initialize process groups for DP and PP. Timeout configurable via ELF_TIMEOUT env var.
		"""
		rank = dist.get_rank()
		world_size = dist.get_world_size()

		timeout = os.getenv("ELF_TIMEOUT", None)
		if timeout is not None:
			timeout = timedelta(seconds=int(timeout))

		self.pp_group = None
		self.dp_groups = []
		for pp in range(self.dp):
			members = [r for r in range(world_size) if r // self.pp == pp]
			if rank == members[0]:
				logger.debug(f"Creating PP group with members {members}")
			pp_group = dist.new_group(members, timeout=timeout)
			if pp == self.dp_rank:
				self.pp_group = pp_group

		if self.dp > 1:
			for dp in range(self.pp):
				members = [r for r in range(world_size) if r % self.pp == dp]
				if rank == members[0]:
					logger.debug(f"Creating DP group with members {members}")
				self.dp_groups.append(dist.new_group(members, timeout=timeout))

	def _init_p2p_process_groups(self, signatures):
		"""
		Initialize point-to-point process groups for efficient communication.
		By default, torch distributed uses 1 cuda stream per pair of ranks.
		Bidirectional communications are therefore sequentialized on that stream, even if they are independent.
		We create separate groups for every type of communication (recv / send * fwd / bwd).

		:param signatures: Dataflow signatures for each block
		:type signatures: List[Signature]
		:return: Dictionary mapping block_id to (recv_pgs, send_pgs) for each block this rank owns
		:rtype: Dict[int, Tuple[Dict[int, ProcessGroup], Dict[int, ProcessGroup]]]
		"""
		rank = dist.get_rank()
		device = (
			torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else None
		)

		block_p2p_groups = {}

		# Get block IDs this rank owns
		ids = self.placement.get_ids(rank)

		# Gather all sources and targets for this rank's blocks
		# pairs are BLOCKS, not RANKS
		local_recv_pairs = []
		local_send_pairs = []
		for block_id in ids:
			signature = signatures[block_id]
			sources = [sig for sig in signature.get_all_sources() if sig is not None]
			targets = [tgt for tgt in signature.get_all_targets() if tgt is not None]
			local_recv_pairs.extend([(block_id, src) for src in sources])
			local_send_pairs.extend([(block_id, dst) for dst in targets])

		local_pairs = local_recv_pairs + local_send_pairs
		all_pairs_list = [None] * self.pp
		dist.all_gather_object(all_pairs_list, local_pairs, group=self.pp_group)

		# Flatten and deduplicate
		all_pairs = set()
		for pairs in all_pairs_list:
			for pair in pairs:
				# Normalize pairs so (a, b) and (b, a) are treated the same
				normalized = tuple(sorted(pair))
				all_pairs.add(normalized)

		# Sort for deterministic ordering across all ranks
		all_pairs = sorted(list(all_pairs))

		# Create all groups, but only save the ones each block needs
		for block_id in ids:
			signature = signatures[block_id]
			sources = [sig for sig in signature.get_all_sources() if sig is not None]
			targets = [tgt for tgt in signature.get_all_targets() if tgt is not None]

			recv_pgs = {"fwd": {}, "bwd": {}}
			send_pgs = {"fwd": {}, "bwd": {}}

			for pair in all_pairs:
				block_a, block_b = pair
				rank_a, rank_b = (
					self.placement[block_a],
					self.placement[block_b],
				)  # map to actual ranks only for PG creation
				group_fwd = dist.new_group(
					[rank_a, rank_b], device_id=device
				)  # might fail if use_local_synchronization is not supported (for instance with MPI)
				group_bwd = dist.new_group([rank_a, rank_b], device_id=device)

				# Save group if this block needs it
				if block_a in ids and block_b in sources:
					recv_pgs["fwd"][block_b] = group_fwd
					send_pgs["bwd"][block_b] = group_bwd
				elif block_a in ids and block_b in targets:
					send_pgs["fwd"][block_b] = group_fwd
					recv_pgs["bwd"][block_b] = group_bwd
				# Reverse direction
				elif block_b in ids and block_a in sources:
					recv_pgs["fwd"][block_a] = group_fwd
					send_pgs["bwd"][block_a] = group_bwd
				elif block_b in ids and block_a in targets:
					send_pgs["fwd"][block_a] = group_fwd
					recv_pgs["bwd"][block_a] = group_bwd

			block_p2p_groups[block_id] = (recv_pgs, send_pgs)

		return block_p2p_groups

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
			parts, signatures = partition_graph(
				model, len(self.placement), sample, partitioner=self.partitioner, tracer=self.tracer
			)
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
