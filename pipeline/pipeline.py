"""
Pipeline API and setup
"""

import os
import torch
import torch.distributed as dist
from .block import PipelineBlock
from .schedule import *
from .engine import Engine
from .utils import TensorMetadata, NameMapping
from .task_graph import graph_from_schedule, find_cycles, fix_cycle
from .partitioners import partition_graph, get_inputs_outputs_single, create_subgraph

import logging

logger = logging.getLogger("pipeline")


class Pipeline:
	"""
	Model wrapper for pipelining that manages the pipeline setup
	"""

	def __init__(self, model, sample, placement="auto", partition="metis", schedule="afab"):
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
		"""
		if not dist.is_initialized() or "RANK" not in os.environ.keys():
			logger.warning(
				"Trying to create a pipeline but no multi-gpu distributed setup has been found."
			)

		ws = int(os.getenv("WORLD_SIZE"))
		if placement == "auto":
			placement = [i for i in range(ws)]

		assert max(placement) < ws, "Placement is out of bounds"

		if isinstance(partition, str):
			try:
				model, inputs, outputs = shared_partition(model, placement, sample, partition)
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

		self._register_metadata(batch)

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
			losses = torch.tensor(losses, device=torch.cuda.current_device())
			losses = losses.sum(dim=0, keepdim=True)
			return result, losses
		else:
			return None, None

	def parameters(self):
		return [{"params": block.model.parameters()} for block in self.blocks]

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

	def _register_metadata(self, batch):
		"""
		Register metadata for input tensors.

		This method is called before the first forward pass to register the metadata
		(shape and dtype) of input tensors. This information is used to allocate
		tensors for communication later in the pipeline.

		:param batch: Input batch of tensors
		:type batch: List[torch.Tensor]
		"""
		# Full forward pass to register metadata used to allocate tensors later
		if self.blocks[0].previous is None:
			# Take all
			for k, v in zip(self.blocks[0].inputs, batch):
				self.blocks[0].metadata[k] = TensorMetadata(v[0])  # Don't register batch size

		for i in range(len(self.blocks)):
			b = self.blocks[i]
			# Sync metadata of fused blocks
			if i > 0 and self.blocks[i - 1].id == b.id - 1:
				mapping = NameMapping(b.inputs, self.blocks[i - 1].outputs)
				b.name_mapping_in = mapping
				self.blocks[i - 1].name_mapping_out = mapping
				b.metadata = {mapping.to_input(k): v for k, v in self.blocks[i - 1].out_metadata.items()}
				logger.debug(f"Synced metadata of {b} and {self.blocks[i - 1]} : {mapping}")

			b.register_metadata()

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
		rank = int(os.getenv("RANK")) if "RANK" in os.environ.keys() else "cpu"

		ids = [i for i in range(len(self.placement)) if self.placement[i] == rank]
		blocks = []
		for i in range(len(layers)):
			new_block = PipelineBlock(layers[i], ids[i], self.placement, inputs[i], outputs[i])
			blocks.append(new_block)
			logger.info(f"{new_block} : inputs = {new_block.inputs}, outputs = {new_block.outputs}")
		return blocks


def shared_partition(model, placement, sample, mode):
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

	# Rank 0 profiles & partition the graph, then shares it to everyone
	# TODO: what if devices are heterogenous ? how to profile correctly ?
	input_list = None
	if rank == 0:
		assert mode in [
			"naive",
			"constrained",
			"dagP",
			"metis",
		], "Partition strategies available are : [naive, constrained, dagP, metis]"

		blocks, inputs, outputs = partition_graph(model, len(placement), sample, mode=mode)
		partition = list(zip(blocks, inputs.values(), outputs.values()))
		input_list = [[] for _ in range(max(placement) + 1)]
		for i, p in enumerate(placement):
			input_list[p].append(partition[i])

	output_list = [None]
	dist.scatter_object_list(output_list, input_list, src=0)
	model, inputs, outputs = (
		[m.cuda() for m, _, _ in output_list[0]],
		[i for _, i, _ in output_list[0]],
		[o for _, _, o in output_list[0]],
	)
	return model, inputs, outputs


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
