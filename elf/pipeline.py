"""
Pipeline API and orchestration
"""

import os

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist

from .execution import PipelineBlock, Engine
from .scheduling import schedule_to_str, check_schedule_validity, reorder_communications
from .utils import send_models, recv_models, broadcast_models, Placement
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
	tracer: Tracer | str = "default"
	partitioner: Partitioner | str = "constrained"
	scheduler: Scheduler | str = "1f1b"
	placement: Placement | str = "auto"
	pp: int | None = None
	dp: int = 1
	worker: int = 0

	def to_kwargs(self) -> dict[str, Any]:
		return {k: v for k, v in asdict(self).items() if v is not None}


class Pipeline:
	"""
	Model wrapper for pipelining that manages the pipeline setup and API
	"""

	def __init__(
		self,
		model,
		sample=None,
		# Legacy arguments
		placement="auto",
		partitioner="constrained",
		scheduler="1f1b",
		dp=1,
		worker=0,
		sources=None,
		targets=None,
		# New config usage
		*,
		config: PipelineConfig | None = None,
	):
		"""
		:param model: the entire model to pipeline, or this rank's portion of a pre-partitioned model
		:type model: nn.Module
		:param sample: sample inputs used for profiling. Not needed when using pre-partitioned model.
		:type sample: torch.Tensor or List[torch.Tensor]
		:param placement: list of device ranks. Block ``i`` of the pipeline will be placed on rank ``placement[i]``. Leave to default ("auto") for automatic placement.``
		:type placement: List[int] or str
		:param partitioner: if your model is already partitioned, set to False. Otherwise set to the partition strategy you want to use (default = metis), which will try to create balanced blocks according to their profiled execution time.
		:type partitioner: boolean or str
		:param scheduler: static scheduling algorithm to use. currently supported : GPipe ("afab"), PipeDream ("1f1b") (default), Hanayo ("hanayo"), ZBH1/ZBH2/ZBV ("zbh1", "zbh2", "zbv"), Full Remat ("full_remat"), Inference ("inference"). You can also define your own scheduler by registering it in the registry.
		:type scheduler: str or function(List[int], int, **kwargs) -> List[Operation]
		:param dp: number of data parallel processes to use.
		:type dp: Optional[int]
		:param worker: rank of the process that will profile the model and partition it
		:type worker: int
		:param sources: For each stage of the entire model (not only this rank's portion), source block id for each input variable. Only needed when partitioner is False, i.e. your model is already partitioned. See :pyfunc:`partitioners.utils.get_sources_targets_sequential` for an example.
		:type sources: List[Dict[str, int]]
		:param targets: For each stage of the entire model (not only this rank's portion), target block ids for each output variable. Only needed when partitioner is False, i.e. your model is already partitioned. See :pyfunc:`partitioners.utils.get_sources_targets_sequential` for an example.
		:type targets: List[Dict[str, List[int]]]
		"""
		if not dist.is_initialized():
			logger.warning(
				"Trying to create a pipeline but no multi-gpu distributed setup has been found."
			)
		ws = dist.get_world_size()

		cfg: dict[str, Any] = {}
		if config is not None:
			cfg = config.to_kwargs()
			cfg.setdefault("placement", placement)
			cfg.setdefault("partitioner", partitioner)
			cfg.setdefault("scheduler", scheduler)
			cfg.setdefault("dp", dp)
			cfg.setdefault("worker", worker)
		else:
			cfg = {
				"placement": placement,
				"partitioner": partitioner,
				"scheduler": scheduler,
				"tracer": "default",  # hardcoded for now
				"dp": dp,
				"worker": worker,
			}

		# Create config
		self.cfg = PipelineConfig(**cfg)
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

	def __call__(self, batch, target, loss_fn, split_size=0, profile=False):
		"""
		Execute the schedule on a batch of data

		:param batch: input data
		:type batch: torch.Tensor or List[torch.Tensor]
		:param target: targets
		:type target: torch.Tensor
		:param loss_fn: loss function to be used. We recommend using torch's built-in loss functions, but you can pass any function that matches the signature. Be careful, the loss is computed on each micro-batch, then averaged over the batch dimension. Depending on the loss function, this may not be equivalent to computing the loss on the full batch.
		:type loss_fn: Function (Tensor,Tensor) -> Tensor
		:param split_size: either one size for equal micro batches (last one may be smaller if the batch size is not divisible by the split size), or a list of possibly different micro batch sizes. In that case the sum of the sizes must be equal to the batch size. Default value is (batch_size // number of gpus)
		:type split_size: int or List[int]
		:param profile: Whether to activate nvidia profiling or not. If True, NVTX ranges will be generated for each operation
		:type profile: boolean

		:return: result of the forward pass and loss value if the last block of the pipeline is managed by this process
		:rtype: (List[Tensor], Tensor) or (None, None)
		.. warning::
			The result is automatically offloaded to CPU. Since merging it back would cause a sync point, it is currently returned as a list of tensors, one for each micro-batch. Avoid merging it back to a single tensor between iterations if you want to avoid performance issues.
		"""
		# We expect a list of arguments, not a single tensor
		if isinstance(batch, torch.Tensor):
			batch = [batch]

		# Move to GPU
		batch = tuple(item.cuda() if isinstance(item, torch.Tensor) else item for item in batch)
		if target is not None:  # target can be None for inference
			target = target.cuda()  # We don't support multiple targets yet

		mb_sizes = self._get_mb_sizes(split_size, batch)
		n_micro_batches = len(mb_sizes)

		if n_micro_batches != self.last_nmb:
			# We have to recompute the schedule
			self._generate_schedule(n_micro_batches)

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

			for name, module in block.model.named_modules():
				if isinstance(module, LayerDW):
					if not module.is_empty("input") or not module.is_empty("grad_output"):
						logger.warning(
							f"{block} - Module {name} still has {module._state('input')} values in queues"
						)
					module.clear()

		self.stats = stats
		self.detailed_stats = detailed_stats

		# Merge back the micro-batches outputs/losses into one batch
		if len(result) != 0:
			# Careful! if the result was offloaded to CPU, then this creates a sync point that slows down the execution
			with torch.no_grad():
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

	def parameters(self):
		"""
		Returns an iterator over the parameters of all blocks in the pipeline.

		:return: An iterator yielding parameters for all blocks.
		:rtype: Iterator[torch.nn.Parameter]
		"""
		for block in self.blocks:
			for param in block.model.parameters():
				yield param

	def named_parameters(self):
		"""
		Returns an iterator over the named parameters of all blocks in the pipeline.

		:return: An iterator yielding tuples of (name, parameter) for all parameters in the pipeline.
		:rtype: Iterator[Tuple[str, torch.nn.Parameter]]
		"""
		for block in self.blocks:
			for name, param in block.model.named_parameters():
				yield name, param

	def zero_grad(self, set_to_none=True):
		"""
		Sets the gradients of all parameters in the pipeline to zero.

		:param set_to_none: If True, set the gradients to None instead of zero. This can provide memory savings.
		:type set_to_none: bool
		"""
		for block in self.blocks:
			block.model.zero_grad(set_to_none=set_to_none)

	def clear(self):
		"""
		Clear the pipeline's internal state and destroy process groups.
		"""
		torch.cuda.synchronize()
		dist.barrier()
		for block in self.blocks:
			if block.dp_group:
				logger.debug(
					f"Destroying DP group with members {dist.get_process_group_ranks(block.dp_group)}"
				)
				dist.destroy_process_group(block.dp_group)

		logger.debug(
			f"Destroying PP group with members {dist.get_process_group_ranks(self.blocks[0].pp_group)}"
		)
		dist.destroy_process_group(self.blocks[0].pp_group)

	def checkpoint(self, epoch="init", dir_path="./"):
		"""
		Save the model's state dictionaries to disk.
		One file will be created per rank
		"""
		rank = dist.get_rank()
		dp = rank // self.pp
		pp = rank % self.dp
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
		Save the model's state dictionary to a file.

		.. warning::
			This method should not be called when the pipeline was initialized with `partitioner=False`.

		:param path: The file path where the model state will be saved.
		:type path: str
		:param worker: The rank of the worker that will save the file. Defaults to 0.
		:type worker: int, optional
		"""
		rank = dist.get_rank()
		pp_group = self.blocks[0].pp_group
		if rank == worker:
			full_state = {}
			n_devices = max(self.placement) + 1
			for d in range(n_devices):
				if d == worker:
					param_list = self.named_parameters()
					for p_name, p in param_list.items():
						full_state[p_name] = p.cpu().detach()
				else:
					param_list = [{}]
					dist.recv_object_list(param_list, src=d, group=pp_group)

					for p_name, p in param_list[0].items():
						full_state[p_name] = p.cpu().detach()

			torch.save(full_state, path)
			logger.info(f"Saved model to {path}")

		else:
			if worker in dist.get_process_group_ranks(pp_group):
				dist.send_object_list([self.named_parameters()], dst=worker, group=pp_group)

	def gather_parameters(self, dst=0):
		"""
		Collects the parameters from all ranks and returns a dictionary of tensors.
		"""
		rank = dist.get_rank()
		pp_group = self.blocks[0].pp_group
		my_params = {}
		for block in self.blocks:
			for name, param in block.model.named_parameters():
				if name not in my_params:
					my_params[name] = param.data.detach()
				else:
					logger.warning(
						f"Duplicate parameter {name} found in block {block.id}, using first occurrence"
					)

		all_params = [None] * dist.get_world_size() if rank == dst else None
		dist.gather_object(my_params, all_params, dst=dst, group=pp_group)

		state_dict = {}
		if rank == dst:
			for params in all_params:
				state_dict.update(params)

		return state_dict

	def _get_mb_sizes(self, split_size, batch):
		"""
		Determine the sizes of micro-batches based on the given split_size and batch.

		:param split_size: Either an int for equal-sized micro-batches or a list of ints for custom sizes.
		:type split_size: int or List[int]
		:param batch: The input batch to be split.
		:type batch: List[torch.Tensor]
		:return: A list of micro-batch sizes.
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

	def _generate_schedule(self, n_micro_batches):
		"""
		Generate a schedule for one execution.
		Also reorders communications to avoid deadlocks and improve performance.
		The resulting schedule only contains operations for the current process.

		:param n_micro_batches: Number of micro-batches
		:type n_micro_batches: int
		:return: Generated schedule
		:rtype: List[Operation]
		"""
		schedule = self.scheduler(self.placement, n_micro_batches, self.signatures)
		check_schedule_validity(schedule)

		schedule = reorder_communications(schedule, strategy="smart")

		if dist.get_rank() == 0:
			logger.info(f"Schedule:\n{schedule_to_str(schedule)}")
			logger.debug(f"Schedule:\n{schedule_to_str(schedule, print_comms=True)}")

		# Remove all operations that are not ours
		ids = list(
			map(lambda b: b.id, self.blocks)
		)  # funny python tips: a map in itself can be iterated only once ! never forget to create a list from it before anything else
		self.schedule = list(filter(lambda op: op.block_id in ids, schedule))

	def _partition_model(self, model, sample, sources, targets):
		"""
		Either partitions the model, or makes sure it's already partitioned
		"""
		if not self.partitioner:
			# Model is already the right part, just make sure it's a list
			if isinstance(model, torch.nn.Module):
				parts = [model]
			else:
				parts = model

			if sources is None and targets is None:
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
		Transforms a list of layers placed on different devices to a working pipeline

		:param parts: List of model parts coming from a partitioned model
		:type parts: List[nn.Module]
		:param signatures: List of signatures for each block
		:type signatures: List[Signature]

		:return: list of blocks handled by this process with everything set up for the pipeline to work
		:rtype: List[PipelineBlock]
		"""
		rank = dist.get_rank()

		offset = (rank // self.pp) * self.pp
		placement = Placement([p + offset for p in self.placement])

		ids = placement.get_ids(rank)
		blocks = []
		for i in range(len(parts)):
			dp_group = self.dp_groups[ids[i] % self.pp] if self.dp > 1 else None
			new_block = PipelineBlock(
				parts[i], ids[i], placement, signatures[ids[i]], pp_group=self.pp_group, dp_group=dp_group
			)
			blocks.append(new_block)

		dist.barrier(self.pp_group)  # init communicators for this process group with a collective

		return blocks

	def _init_process_groups(self):
		"""
		Initialize process groups for data parallelism (DP) and pipeline parallelism (PP).
		"""
		rank = dist.get_rank()
		world_size = dist.get_world_size()

		dp_rank = rank // self.pp

		self.pp_group = None
		self.dp_groups = []
		for pp in range(self.dp):
			members = [r for r in range(world_size) if r // self.pp == pp]
			if rank == members[0]:
				logger.debug(f"Creating PP group with members {members}")
			pp_group = dist.new_group(members)
			if pp == dp_rank:
				self.pp_group = pp_group

		if self.dp > 1:
			for dp in range(self.pp):
				members = [r for r in range(world_size) if r % self.pp == dp]
				if rank == members[0]:
					logger.info(f"Creating DP group with members {members}")
				self.dp_groups.append(dist.new_group(members))

	def _shared_partition(self, model, sample):
		"""
		Partitions a model according to a placement & mode, then shares it to every process to be consistent

		:param model: model to partition
		:type model: nn.Module
		:param sample: example of input data that will be processed by the model
		:type sample: Tensor

		:return: blocks (modules) for this process
		:rtype: List[nn.Module]
		"""
		rank = dist.get_rank()

		# Rank 'worker' profiles & partition the graph, then shares it to everyone
		n_blocks = self.placement.count(rank % self.pp)
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
