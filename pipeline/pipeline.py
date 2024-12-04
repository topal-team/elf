"""
Pipeline API and setup
"""

import os
from pipeline.partitioners.utils import Signature
import torch
import shutil
import torch.distributed as dist
from .block import PipelineBlock
from .schedules import *
from .engine import Engine
from .utils import *
from .scheduling import mark_batched_comms
from .partitioners import partition_graph
from collections import OrderedDict

import logging

logger = logging.getLogger("pipeline")


def to_cpu(nested_dict):
	"""
	Recursively moves all tensors in a nested dictionary to the CPU.
	"""
	if isinstance(nested_dict, dict):
		return {key: to_cpu(value) for key, value in nested_dict.items()}
	elif isinstance(nested_dict, list):
		return [to_cpu(item) for item in nested_dict]
	elif isinstance(nested_dict, tuple):
		return tuple(to_cpu(item) for item in nested_dict)
	elif isinstance(nested_dict, torch.Tensor):
		return nested_dict.cpu()
	else:
		return nested_dict


class Pipeline:
	"""
	Model wrapper for pipelining that manages the pipeline setup
	"""

	def __init__(
		self,
		model,
		sample,
		placement="auto",
		partitioner="metis",
		schedule="1f1b",
		dp=1,
		worker=0,
		sources=None,
		targets=None,
	):
		"""
		:param model: the entire model to pipeline, or this rank's portion of a pre-partitioned model
		:type model: nn.Module
		:param sample: sample inputs used for profiling. Not needed when using pre-partitioned model.
		:type sample: torch.Tensor or List[torch.Tensor]
		:param placement: list of device ranks. Block ``i`` of the pipeline will be placed on rank ``placement[i]``. Leave to default ("auto") for automatic placement, which is ``[0, 1, .., world size - 1]``
		:type placement: List[int] or str
		:param partitioner: if your model is already partitioned, set to False. Otherwise set to the partition strategy you want to use (default = metis), which will try to create balanced blocks according to their profiled execution time.
		:type partitioner: boolean or str
		:param schedule: pipeline algorithm to use. currently supported : GPipe ("afab"), PipeDream ("1f1b") (default), Hanayo ("hanayo"). You can also define your own function to generate the schedule, see the existing functions in schedule for an example.
		:type schedule: str or function(List[int], int, **kwargs) -> List[Operation]
		:param dp: number of data parallel processes to use.
		:type dp: Optional[int]
		:param worker: rank of the process that will profile the model and partition it
		:type worker: int
		:param sources: For each stage of the entire model (not only this rank's portion), source block id for each input variable. Only needed when partitioner is False, i.e. your model is already partitioned.
		:type sources: List[Dict[str, int]]
		:param targets: For each stage of the entire model (not only this rank's portion), target block ids for each output variable. Only needed when partitioner is False, i.e. your model is already partitioned.
		:type targets: List[Dict[str, List[int]]]
		"""
		if not dist.is_initialized() or "RANK" not in os.environ.keys():
			logger.warning(
				"Trying to create a pipeline but no multi-gpu distributed setup has been found."
			)
		ws = dist.get_world_size()
		local_rank = int(os.getenv("LOCAL_RANK"))
		torch.cuda.set_device(local_rank)

		if placement == "auto":
			pp = ws // dp
			placement = self._get_default_placement(schedule, pp)
		elif isinstance(placement, str):
			placement = list(map(int, placement.split(",")))

		assert max(placement) < ws, "Placement is out of bounds"
		pp = max(placement) + 1

		assert pp * dp <= ws, f"Requested PP = {pp}, DP = {dp} but only {ws} processes were spawned"

		self.pp = pp
		self.dp = dp

		self._init_process_groups()  # sets pp_group and dp_groups

		self.placement = placement
		parts, signatures = self._partition_model(model, partitioner, sample, worker, sources, targets)
		self.blocks = self._create_pipeline(parts, signatures)
		self.signatures = signatures
		self.scheduler = self._get_scheduler(schedule)
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
		:type loss_fn: Function (Tensor, Tensor) -> Tensor
		:param split_size: either one size for equal micro batches (last one may be smaller if the batch size is not divisible by the split size), or a list of possibly different micro batch sizes. In that case the sum of the sizes must be equal to the batch size. Default value is (batch_size // number of gpus)
		:type split_size: int or List[int]
		:param profile: Whether to activate nvidia profiling or not. If True, NVTX ranges will be generated for each operation
		:type profile: boolean

		:return: result of the forward pass and loss value if the last block of the pipeline is managed by this process
		:rtype: (Tensor, Tensor) or (None, None)
		"""
		# We expect a list of arguments, not a single tensor
		if isinstance(batch, torch.Tensor):
			batch = [batch]

		# Move to GPU
		batch = tuple(item.cuda() if isinstance(item, torch.Tensor) else item for item in batch)
		target = target.cuda()  # We don't support multiple targets yet

		mb_sizes = self._get_mb_sizes(split_size, batch)
		n_micro_batches = len(mb_sizes)

		if n_micro_batches != self.last_nmb:
			# We have to recompute the schedule
			self._generate_schedule(n_micro_batches)

			self.last_nmb = n_micro_batches

		# Execute the schedule
		result, losses, times = self.engine.train_step(
			batch, target, loss_fn, self.schedule, mb_sizes, profile
		)

		# First and last block have some remaining tensors
		for block in self.blocks:
			for var in block.inputs:
				var.clear()
			for var in block.outputs:
				for dst in var:
					dst.clear()

		self.times = times

		# Merge back the micro-batches outputs/losses into one batch
		if len(result) != 0:
			result = torch.cat(result, dim=0)
			losses = torch.tensor(losses, device=result.device)
			losses = losses.sum() / sum(mb_sizes)
			if self.dp > 1:
				dist.all_reduce(losses, group=self.blocks[-1].dp_group, op=dist.ReduceOp.AVG)
			return result, losses
		else:
			return None, None

	def parameters(self):
		"""
		Returns an iterator over the parameters of all blocks in the pipeline.

		:return: An iterator yielding parameter groups for each block.
		:rtype: List[Dict[str, Iterator[torch.nn.Parameter]]]
		"""
		return [{"params": block.model.parameters()} for block in self.blocks]

	def named_parameters(self):
		"""
		Returns an iterator over the named parameters of all blocks in the pipeline.

		:return: An iterator yielding tuples of (name, parameter) for all parameters in the pipeline.
		:rtype: Iterator[Tuple[str, torch.nn.Parameter]]
		"""
		rank_named_parameters = OrderedDict()
		for block in self.blocks:
			rank_named_parameters.update(block.model.named_parameters())
		return rank_named_parameters

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

	def _get_default_placement(self, schedule, pp):
		if schedule == "hanayo":
			return [i for i in range(pp)] + list(reversed([i for i in range(pp)]))
		else:
			return [i for i in range(pp)]

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

	def _get_scheduler(self, schedule):
		if not isinstance(schedule, str):
			return schedule
		match schedule.lower():
			case "afab":
				return generate_afab_schedule
			case "1f1b":
				return generate_1f1b_schedule
			case "hanayo":
				return generate_hanayo_schedule
			case "full_remat":
				return generate_full_remat_schedule
			case _:
				raise Exception(
					f"Unknown schedule : {schedule}. Available ones : [afab, 1f1b, hanayo, full_remat]"
				)

	def _generate_schedule(self, n_micro_batches):
		"""
		Generate a schedule for one execution.

		:param n_micro_batches: Number of micro-batches
		:type n_micro_batches: int
		:param options: Additional options for the scheduler
		:type options: dict
		:return: Generated schedule
		:rtype: List[Operation]
		"""
		schedule = self.scheduler(self.placement, n_micro_batches, self.signatures)

		mark_batched_comms(schedule, self.placement)

		# Remove all operations that are not ours
		ids = list(
			map(lambda b: b.id, self.blocks)
		)  # funny python tips: a map in itself can be iterated only once ! never forget to create a list from it before anything else
		self.schedule = list(filter(lambda op: op.block_id in ids, schedule))

	def _partition_model(self, model, partitioner, sample, worker, sources, targets):
		"""
		Either partitions the model, or makes sure it's already partitioned
		"""
		if isinstance(partitioner, str):
			try:
				parts, signatures = self._shared_partition(model, sample, partitioner, worker=worker)
			except Exception as e:
				logger.error(
					"Error partitioning the model. This can be due to your model using features either not supported by torch.fx/torch.export, or by this library."
				)
				raise e
		elif not partitioner:
			# Model is already the right part, just make sure it's a list
			if isinstance(model, torch.nn.Module):
				parts = [model]
			else:
				parts = model

			assert (
				sources is not None and targets is not None
			), "Sources and targets must be provided when using pre-partitioned model"
			signatures = []
			for i in range(len(self.placement)):
				inputs = sorted(list(sources[i].keys()))
				outputs = sorted(list(targets[i].keys()))
				signatures.append(
					Signature(
						inputs, outputs, [sources[i][j] for j in inputs], [targets[i][j] for j in outputs]
					)
				)

		else:
			raise Exception(
				"Partition strategy should be either False when using pre-partitioned model, or a string among [naive, constrained, dagP, metis]."
			)

		return parts, signatures

	def _create_pipeline(self, parts, signatures):
		"""
		Transforms a list of layers placed on different devices to a working pipeline

		:param parts: List of model parts coming from a partitioned model
		:type parts: List[nn.Module]

		:return: list of blocks handled by this process with everything set up for the pipeline to work
		:rtype: List[PipelineBlock]
		"""
		rank = dist.get_rank()

		offset = (rank // self.pp) * self.pp
		placement = [p + offset for p in self.placement]

		ids = [i for i in range(len(placement)) if placement[i] == rank]
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

	def _shared_partition(self, model, sample, partitioner, worker=0):
		"""
		Partitions a model according to a placement & mode, then shares it to every process to be consistent

		:param model: model to partition
		:type model: nn.Module
		:param sample: example of input data that will be processed by the model
		:type sample: Tensor
		:param partitioner: partitioner to use ; available options are :

			- "naive": simple load balancing algorithm
			- "constrained": naive with less communication
			- "metis": use METIS
			- "dagP": use dagP / rMLGP
			For more info, see partition.

		:type partitioner: str
		:param worker: rank of the process that will profile the model and partition it
		:type worker: int

		:return: blocks (modules) for this process
		:rtype: List[nn.Module]
		"""
		rank = dist.get_rank()

		# Rank 'worker' profiles & partition the graph, then shares it to everyone
		n_blocks = self.placement.count(rank % self.pp)
		blocks = [None for _ in range(n_blocks)]
		signatures = [None for _ in range(len(self.placement))]
		if rank == worker:
			partitioner = self._check_for_partitioner(partitioner)

			parts, signatures = partition_graph(
				model, len(self.placement), sample, partitioner=partitioner
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

		elif rank < self.pp:
			# First pipeline share to their DP replicas
			recv_models(blocks, src=worker, group=self.pp_group)
			dist.recv_object_list(signatures, src=worker, group=self.pp_group)

			torch.cuda.synchronize()

		if self.dp > 1:
			r = rank % self.pp
			broadcast_models(blocks, src=r, group=self.dp_groups[r])
			dist.broadcast_object_list(signatures, src=r, group=self.dp_groups[r])
			torch.cuda.synchronize()

		logger.debug(f"Rank {rank} has {len(blocks)} block" + ("s" if len(blocks) > 1 else ""))

		return blocks, signatures

	def _check_for_partitioner(self, partitioner):
		assert partitioner in [
			"naive",
			"constrained",
			"dagP",
			"metis",
		], "Partition strategies available are : [naive, constrained, dagP, metis]"

		if partitioner == "metis":
			if not shutil.which("gpmetis"):
				logger.warning("metis is not installed, falling back to naive")
				return "naive"
		elif partitioner == "dagP":
			if not shutil.which("rMLGP"):
				logger.warning("dagP is not installed, falling back to metis")
				return self._check_for_partitioner("metis")

		return partitioner


def get_sources_targets_sequential(placement):
	"""
	Generates sources and targets for a fully sequential model (no skip connections), with one input and one output per stage.

	:param placement: placement of the model blocks on gpus
	:type placement: List[int]
	:return: Sources and targets for each stage
	:rtype: Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, List[int]]]]
	"""
	sources = {}
	targets = {}
	for i in range(len(placement)):
		# Everyone needs full signatures to generate schedule
		sources[i] = {"input": i - 1 if i != 0 else None}
		targets[i] = {"output": [i + 1 if i != len(placement) - 1 else None]}
	return sources, targets
