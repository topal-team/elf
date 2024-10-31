"""
Pipeline API and setup
"""

import os
import torch
import torch.distributed as dist
from .block import PipelineBlock
from .schedule import *
from .engine import Engine
from .task_graph import graph_from_schedule, find_cycles, fix_cycle
from .partitioners import partition_graph, get_inputs_outputs_single, create_subgraph

import logging

logger = logging.getLogger("pipeline")


class Pipeline:
	"""
	Model wrapper for pipelining that manages the pipeline setup
	"""

	def __init__(
		self, model, sample, placement="auto", partition="naive", schedule="afab", dp=1, worker=0
	):
		"""
		:param model: the entire model to pipeline, or this rank's portion of a pre-partitioned model
		:type model: nn.Module
		:param sample: sample inputs used for profiling. Not needed when using pre-partitioned model.
		:type sample: torch.Tensor or List[torch.Tensor]
		:param placement: list of device ranks. Block ``i`` of the pipeline will be placed on rank ``placement[i]``. Leave to default ("auto") for automatic placement, which is ``[0, 1, .., world size - 1]``
		:type placement: List[int] or str
		:param partition: if your model is already partitioned, set to False. Otherwise set to the partition strategy you want to use (default = metis), which will try to create balanced blocks according to their profiled execution time.
		:type partition: boolean or str
		:param schedule: pipeline algorithm to use. currently supported : GPipe ("afab") (default), PipeDream ("1f1b"), Hanayo ("hanayo"). You can also define your own function to generate the schedule, see the existing functions in schedule for an example.
		:type schedule: str or function(List[int], int, **kwargs) -> List[Operation]
		:param dp: number of data parallel processes to use.
		:type dp: Optional[int]
		"""
		if not dist.is_initialized() or "RANK" not in os.environ.keys():
			logger.warning(
				"Trying to create a pipeline but no multi-gpu distributed setup has been found."
			)
		ws = dist.get_world_size()
		local_rank = int(os.getenv("LOCAL_RANK"))
		torch.cuda.set_device(local_rank)

		if placement == "auto":
			placement = self._get_default_placement(schedule, pp)

		assert max(placement) < ws, "Placement is out of bounds"
		pp = max(placement) + 1

		assert pp * dp <= ws, f"Requested PP = {pp}, DP = {dp} but only {ws} processes were spawned"

		self.pp = pp
		self.dp = dp

		if isinstance(partition, str):
			try:
				model, inputs, outputs = shared_partition(
					model, placement, sample, partition, worker=worker
				)
			except Exception as e:
				logger.error(
					"Error partitioning the model. This probably means that your model uses features either not supported by torch.fx/torch.export, or by this library (such as skip connections). If the error persists, consider using a pre-partitioned model"
				)
				raise e
		elif not partition:
			try:
				model, inputs, outputs = local_partition(model, placement)
			except Exception as e:
				logger.error(
					"Error partitioning the model. This probably means that your model uses features either not supported by torch.fx/torch.export"
				)
				raise e
		else:
			raise Exception(
				"Partition strategy should be either False when using pre-partitioned model, or a string among [naive, constrained, dagP, metis]."
			)

		self.placement = placement
		self.scheduler = self._get_scheduler(schedule)
		self.blocks = self._create_pipeline(model, inputs, outputs)
		self._init_process_groups()
		self.engine = Engine(self.blocks)

		# Used to avoid re-generating schedule every time
		self.schedule = []
		self.last_options = {}
		self.last_nmb = 0

	def __call__(self, batch, target, loss_fn, split_size=1, profile=False, **options):
		"""
		Execute the schedule on a batch of data

		:param batch: input data
		:type batch: torch.Tensor or List[torch.Tensor]
		:param target: targets
		:type target: torch.Tensor
		:param loss_fn: loss function to be used. We recommend using torch's built-in loss functions, but you can pass any function that matches the signature. Be careful, summing the loss of every micro-batch should be equivalent to computing the loss on the full batch (i.e., no average over batch dimension)
		:type loss_fn: Function (Tensor, Tensor) -> Tensor
		:param split_size: either one size for equal micro batches (last one may be smaller if the batch size is not divisible by the split size), or a list of possibly different micro batch sizes. In that case the sum of the sizes must be equal to the batch size.
		:type split_size: int or List[int]
		:param profile: Whether to activate nvidia profiling or not. If True, NVTX ranges will be generated for each operation
		:type profile: boolean

		:return: result of the forward pass and loss value if the last block of the pipeline is managed by this process
		:rtype: (Tensor, Tensor) or (None, None)
		"""
		# We expect a list of arguments, not a single tensor
		if isinstance(batch, torch.Tensor):
			batch = [batch]

		mb_sizes = self._get_mb_sizes(split_size, batch)
		n_micro_batches = len(mb_sizes)

		if n_micro_batches != self.last_nmb or options != self.last_options:
			# We have to recompute the schedule
			self._generate_schedule(n_micro_batches, options)

			self.last_options = options
			self.last_nmb = n_micro_batches

		# Execute the schedule
		result, losses, times = self.engine.train_step(
			batch, target, loss_fn, self.schedule, mb_sizes, profile
		)

		# First and last block have some remaining tensors
		for block in self.blocks:
			block.grads_to_send.clear()
			block.act_to_send.clear()

		self.times = times

		# Merge back the micro-batches outputs/losses into one batch
		if len(result) != 0:
			result = torch.cat(result, dim=0)
			losses = torch.tensor(losses, device=result.device)
			losses = losses.mean()
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
		full_params = {}
		for block in self.blocks:
			for p_name, p in block.model.named_parameters():
				full_params[p_name] = p
		return full_params

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

	def save(self, path, worker=0):
		"""
		Save the model's state dictionary to a file.

		.. warning::
			This method should not be called when the pipeline was initialized with `partition=False`.

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

		else:
			if worker in dist.get_process_group_ranks(pp_group):
				dist.send_object_list([self.named_parameters()], dst=worker, group=pp_group)

	def _get_default_placement(self, schedule, pp):
		if schedule == "hanayo":
			return [i for i in range(pp)] + reversed([i for i in range(pp)])
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
			n_micro_batches = batch_size // split_size
			mb_sizes = [split_size for _ in range(n_micro_batches)]
			if batch_size % split_size != 0:
				mb_sizes.append(batch_size % split_size)
		else:
			assert sum(split_size) == batch_size, "Splits do not cover the entire batch"
			mb_sizes = split_size

		return mb_sizes

	def _get_scheduler(self, schedule):
		match schedule.lower():
			case "afab":
				return generate_afab_schedule
			case "1f1b":
				return generate_1f1b_schedule
			case "hanayo":
				return generate_hanayo_schedule
			case _:
				return schedule

	def _generate_schedule(self, n_micro_batches, options):
		"""
		Generate a schedule for one execution.

		:param n_micro_batches: Number of micro-batches
		:type n_micro_batches: int
		:param options: Additional options for the scheduler
		:type options: dict
		:return: Generated schedule
		:rtype: List[Operation]
		"""
		schedule = self.scheduler(self.placement, n_micro_batches, **options)

		# Construct graph, detech cycle and add communication batching to fix them
		cycles = find_cycles(graph_from_schedule(schedule))
		if cycles and dist.get_rank() == 0:
			logger.warning("Found potential deadlocks in the schedule ! Fixing them.")
			for c in cycles:
				logger.debug(f"Cycle : {c}")
		for c in cycles:
			fix_cycle(c)

		# Remove all operations that are not ours
		ids = list(
			map(lambda b: b.id, self.blocks)
		)  # funny python tips: a map in itself can be iterated only once ! never forget to create a list from it before anything else
		self.schedule = list(filter(lambda op: op.block_id in ids, schedule))

	def _create_pipeline(self, layers, inputs, outputs):
		"""
		Transforms a list of layers placed on different devices to a working pipeline

		:param layers: List of layers / groups of layers coming from a partitioned model. This list has to be sequential in terms of computation
		:type layers: List[nn.Module]
		:param inputs: name of variables taken as input by each block
		:type inputs: List[List[str]]
		:param outputs: name of variables returned by each block
		:type outputs: List[List[str]]

		:return: list of blocks handled by this process with everything set up for the pipeline to work
		:rtype: List[PipelineBlock]
		"""
		rank = dist.get_rank()

		offset = (rank // self.pp) * self.pp
		placement = [p + offset for p in self.placement]

		ids = [i for i in range(len(placement)) if placement[i] == rank]
		blocks = []
		for i in range(len(layers)):
			new_block = PipelineBlock(layers[i], ids[i], placement, inputs[i], outputs[i])
			blocks.append(new_block)
			logger.info(f"{new_block} : inputs = {new_block.inputs}, outputs = {new_block.outputs}")

		return blocks

	def _init_process_groups(self):
		"""
		Initialize process groups for data parallelism (DP) and pipeline parallelism (PP).
		"""
		rank = dist.get_rank()
		world_size = dist.get_world_size()

		offset = (rank // self.pp) * self.pp
		placement = [p + offset for p in self.placement]

		ids = [i for i in range(len(placement)) if placement[i] == rank]
		if self.dp > 1:
			for stage in range(len(placement)):
				members = [r for r in range(world_size) if (placement[stage] % self.pp) == (r % self.pp)]
				if rank == members[0]:
					logger.info(f"Creating DP group with members {members}")
				dp_group = dist.new_group(members)
				if stage in ids:
					self.blocks[ids.index(stage)].dp_group = dp_group

		if self.pp > 1:
			for dp_rank in range(self.dp):
				members = [r for r in range(world_size) if r // self.pp == dp_rank]
				if rank == members[0]:
					logger.info(f"Creating PP group with members {members}")
				pp_group = dist.new_group(members)
				if rank in members:
					for b in self.blocks:
						b.pp_group = pp_group

					# Init communicators to avoid hangs later on
					buffer = torch.empty(1, device=torch.cuda.current_device())
					dist.all_reduce(buffer, group=pp_group)
					torch.cuda.synchronize()


def shared_partition(model, placement, sample, mode, worker=0):
	"""
	Partitions a model according to a placement & mode, then shares it to every process to be consistent

	:param model: model to partition
	:type model: nn.Module
	:param placement: list of device ranks
	:type placement: List[int]
	:param sample: example of input data that will be processed by the model
	:type sample: Tensor
	:param mode: partitioner to use ; available options are :

		- "naive": simple load balancing algorithm
		- "constrained": naive with less communication
		- "metis": use METIS
		- "dagP": use dagP / rMLGP
		For more info, see partition.

	:type mode: str

	:return:

		- Blocks for this process
		- Inputs for each block of this process
		- Outputs for each block of this process

	:rtype: List[nn.Module], List[List[str]], List[List[str]]
	"""
	rank = dist.get_rank()
	ws = dist.get_world_size()
	n_devices = max(placement) + 1

	# Rank 0 profiles & partition the graph, then shares it to everyone
	# TODO: what if devices are heterogenous ? how to profile correctly ?
	if rank == worker:
		assert mode in [
			"naive",
			"constrained",
			"dagP",
			"metis",
		], "Partition strategies available are : [naive, constrained, dagP, metis]"

		blocks, all_inputs, all_outputs = partition_graph(model, len(placement), sample, mode=mode)
		for d in range(n_devices):
			blocks_on_d = [blocks[i] for i in range(len(blocks)) if placement[i] == d]
			inputs_on_d = [all_inputs[i] for i in all_inputs.keys() if placement[i] == d]
			outputs_on_d = [all_outputs[i] for i in all_outputs.keys() if placement[i] == d]
			if d == worker:
				models, inputs, outputs = blocks_on_d, inputs_on_d, outputs_on_d
			else:
				dist.send_object_list([blocks_on_d, inputs_on_d, outputs_on_d], dst=d)
				# torch.cuda.synchronize()

		for dp in range(1, ws // n_devices):
			dst = rank + dp * n_devices
			dist.send_object_list([models, inputs, outputs], dst=dst)
			# torch.cuda.synchronize()

	elif rank < n_devices:
		blocks = [None for _ in range(3)]

		dist.recv_object_list(blocks, src=worker)
		torch.cuda.synchronize()
		models, inputs, outputs = blocks
		for dp in range(1, ws // n_devices):
			dst = rank + dp * n_devices
			dist.send_object_list(blocks, dst=dst)
			# torch.cuda.synchronize()
	else:
		blocks = [None for _ in range(3)]
		dist.recv_object_list(blocks, src=rank % n_devices)
		torch.cuda.synchronize()
		models, inputs, outputs = blocks

	for m, i, o in zip(models, inputs, outputs):
		logger.info(f"Rank {rank} - signature = {i} -> {o}")
		logger.debug(f"Rank {rank} - code = {m.code}")

	return models, inputs, outputs


def local_partition(model, placement):
	"""
	Partitions a pre-partitioned model locally for each process.

	This function takes a model that has already been partitioned and identifies the model parts
	assigned to the current rank, traces them, and extracts their inputs and outputs.

	:param model: The pre-partitioned model. Can be a single nn.Module or a list of nn.Modules.
	:type model: nn.Module or List[nn.Module]
	:param placement: A list indicating the rank assignment for each part of the model.
	:type placement: List[int]

	:return: A tuple containing:
		- The partitioned model parts for the current rank
		- A list of inputs for each model part
		- A list of outputs for each model part
	:rtype: Tuple[List[nn.Module], List[List[str]], List[List[str]]]
	"""
	rank = dist.get_rank()
	inputs = {}
	outputs = {}
	if isinstance(model, torch.nn.Module):
		model = [model]
	ids = [i for i in range(len(placement)) if placement[i] == rank]
	for i in range(len(ids)):
		trace = torch.fx.symbolic_trace(model[i])
		new, inputs[i], outputs[i] = get_inputs_outputs_single(list(trace.graph.nodes))
		model[i] = create_subgraph(trace, new, inputs[i], outputs[i])

	return model, inputs, outputs
