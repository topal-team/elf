"""
Measuring what a partitioned model costs to run.

Everything downstream of the partition -- the rematerialization ILP, the communication
schedulers, and anything that wants to predict an iteration time -- needs the same handful of
numbers per pipeline stage, so they are measured once, in one pass, and handed around as
:class:`StageProfile`.

This is *stage* profiling, and it only makes sense once the model has been cut into stages and
placed. The partitioner has its own, unrelated measurement pass
(:func:`elf.partitioners.profile.profile_operations`): it works per graph node rather than per
stage, and it necessarily runs before there are any stages to profile.
"""

from collections import defaultdict
import logging
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
import torch.distributed as dist

from torch._subclasses.fake_tensor import FakeTensorMode

from elf.signature import Signature
from elf.utils import PreHide, TensorMetadata, Timer
from elf.zb_utils import LayerDW

__all__ = ["StageProfile", "profile_stage", "profile_pipeline", "init_tensor_"]

logger = logging.getLogger(__name__)


@dataclass
class StageProfile:
	"""
	What one pipeline stage costs, for one micro-batch.

	Every number is measured on the device the stage runs on, for a single micro-batch of the
	size the pipeline was given as a sample. Times are in **milliseconds**, memory sizes in
	**mebibytes**.

	The three operations are the ones the schedule vocabulary knows: the forward pass
	(:py:attr:`~elf.scheduling.OperationType.FORWARD`), the input gradients
	(``BACKWARD_INPUTS``) and the parameter gradients (``BACKWARD_PARAMS``, the *W* of a
	zero-bubble schedule). Recomputation costs nothing new to measure: replaying a forward costs
	:py:attr:`t_forward` again.

	Memory is reported two ways, because rematerialization needs both: ``m_*`` is what the
	operation still holds once it returns, and ``m_peak_*`` is the high-water mark reached while
	it runs, which is what a memory budget has to fit. Both count the stage's inputs exactly when
	the stage keeps them alive, as the graph does when its first operation saves what it reads.
	"""

	#: time of the forward pass
	t_forward: float
	#: time of the input gradients
	t_backward_inputs: float
	#: time of the parameter gradients
	t_backward_params: float
	#: time to send this stage's output to the next one, 0 when the next stage is on this rank
	t_transfer: float

	#: memory still held after the forward pass
	m_forward: float
	#: memory still held after the input gradients
	m_backward_inputs: float
	#: memory still held after the parameter gradients
	m_backward_params: float

	#: peak memory reached during the forward pass
	m_peak_forward: float
	#: peak memory reached during the input gradients
	m_peak_backward_inputs: float
	#: peak memory reached during the parameter gradients
	m_peak_backward_params: float
	#: peak memory of a forward that saves nothing, as full rematerialization runs it
	m_peak_forward_no_grad: float

	#: memory of the stage's parameters
	m_parameters: float
	#: memory of the stage's input tensors
	m_inputs: float
	#: memory of the stage's output tensors
	m_outputs: float
	#: memory of the inputs the backward pass keeps for the parameter gradients
	m_saved_inputs: float

	def to_dict(self) -> dict:
		"""
		The profile as plain numbers, ready to be serialized.

		:rtype: dict
		"""
		return asdict(self)

	@classmethod
	def from_dict(cls, values: dict) -> "StageProfile":
		"""
		Rebuild a profile from :py:meth:`to_dict`, filling anything absent with zero.

		Missing entries are tolerated so that a caller which only has timings -- the schedules
		solved offline are fed exactly that -- can still drive the communication schedulers.

		:param values: a mapping with a subset of the profile's fields
		:type values: dict
		:rtype: StageProfile
		"""
		known = {f.name for f in fields(cls)}
		unknown = set(values) - known
		if unknown:
			raise ValueError(f"Unknown profiling fields: {sorted(unknown)}")
		return cls(**{name: float(values.get(name, 0.0)) for name in known})


def init_tensor_(tensor: torch.Tensor):
	"""Fill tensor in-place with random data appropriate for its dtype.

	.. warning:: For integer dtype, the tensor is filled with random integers between 0 and 100.
		If you have an embedding with dim < 100, this will cause an error.
	"""
	match tensor.dtype:
		case torch.float32 | torch.float16 | torch.bfloat16:
			tensor.normal_(mean=0.0, std=1.0)
		case torch.int32 | torch.int64 | torch.int8 | torch.int16:
			tensor.random_(0, 100)
		case torch.bool:
			tensor.random_(0, 2)
		case _:
			raise ValueError(f"Unsupported dtype: {tensor.dtype}")


def _bparams(stage: torch.nn.Module):
	for module in stage.modules():
		if isinstance(module, LayerDW):
			module.commit(0)
			module.backward(0)


def _saved_input_bytes(stage: torch.nn.Module) -> float:
	"""
	How much of what ``W`` is waiting for is activation rather than gradient.

	A :class:`~elf.zb_utils.LayerDW` records one ``(input, gradient)`` pair per use of it, and only
	the input half can be produced again by recomputing the forward. Read between the backward and
	``W``, while the pairs are still staged and before ``W`` consumes them.

	:rtype: float
	"""
	total = 0
	storages = defaultdict(int)
	for module in stage.modules():
		if isinstance(module, LayerDW):
			# count unique storages
			for input, _ in module.last_pairs or ():
				if input is not None:
					ptr = input.untyped_storage()
					storages[ptr] = (
						ptr.nbytes()
					)  # storage is only dropped when every view is deleted, so counting the entire storage is accurate

	total = sum(v for v in storages.values())
	return total / 1024 / 1024


def _stage_inputs(sample: tuple[torch.Tensor, ...]) -> list[PreHide]:
	"""
	Fresh copies of the sample, set up the way :meth:`~elf.execution.PipelineBlock.forward` sets up
	what it receives.

	Allocating them inside a measurement is what makes the inputs count towards what the stage
	holds, and wrapping them like the block does is what lets them go once nothing but the
	profiler refers to them: only what the graph saves keeps them alive, as in the pipeline.

	:rtype: list[PreHide]
	"""
	inputs = []
	for tensor in sample:
		copy = tensor.clone()
		if copy.is_floating_point() or copy.is_complex():
			copy.requires_grad_(True)
		inputs.append(PreHide(copy))
	return inputs


def _fake_loss(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
	return sum(o.sum() for o in outputs if isinstance(o, torch.Tensor))


def _measure_full_pass(
	stage: torch.nn.Module, sample: tuple[torch.Tensor, ...], n_iter: int = 3
) -> tuple[dict, dict, dict, float, float, float, torch.Tensor]:
	"""Measure time, held memory and peak memory of F/B/W, plus input and output sizes.

	The inputs the backward leaves behind for ``W`` are measured here too, since they only exist
	between the two and this is the pass that runs both.

	The warmup output (detached, on CPU) is returned so that it can be used as the input sample
	for the next pipeline stage, avoiding an extra forward pass just for shape propagation.
	"""
	outputs = stage(*(x.get() for x in _stage_inputs(sample)))
	outputs = (outputs,) if not isinstance(outputs, tuple) else outputs
	warmup_output = outputs[0].detach().cpu()
	loss = _fake_loss(outputs)
	loss.backward()
	_bparams(stage)
	del outputs, loss
	torch.cuda.synchronize()

	times = {"f": [], "b": [], "w": []}
	mems = {"f": [], "b": [], "w": []}
	peaks = {"f": [], "b": [], "w": []}
	saved_inputs = []

	for _ in range(n_iter):
		torch.cuda.reset_peak_memory_stats()
		start_mem = torch.cuda.memory_allocated()
		inputs = _stage_inputs(sample)
		with Timer() as timer:
			outputs = stage(*(x.get() for x in inputs))
			outputs = (outputs,) if not isinstance(outputs, tuple) else outputs
			loss = _fake_loss(outputs)

		minput = sum(s.nbytes for s in sample) / 1024 / 1024
		moutput = (
			sum(output.nbytes for output in outputs if isinstance(output, torch.Tensor)) / 1024 / 1024
		)

		times["f"].append(timer.time() * 1000)  # in ms
		mems["f"].append((torch.cuda.memory_allocated() - start_mem) / 1024 / 1024)
		peaks["f"].append((torch.cuda.max_memory_allocated() - start_mem) / 1024 / 1024)
		torch.cuda.reset_peak_memory_stats()

		before_b = torch.cuda.memory_allocated()
		with Timer() as timer:
			loss.backward()
			del outputs, loss

		# the input gradients are part of the backward's peak, but the block sends them away
		# right after it, so they are not something the backward holds on to
		del inputs

		times["b"].append(timer.time() * 1000)  # in ms
		mems["b"].append((torch.cuda.memory_allocated() - start_mem) / 1024 / 1024)
		peaks["b"].append((torch.cuda.max_memory_allocated() - before_b) / 1024 / 1024)
		saved_inputs.append(_saved_input_bytes(stage))

		before_w = torch.cuda.memory_allocated()
		torch.cuda.reset_peak_memory_stats()
		with Timer() as timer:
			_bparams(stage)

		times["w"].append(timer.time() * 1000)  # in ms
		mems["w"].append((torch.cuda.memory_allocated() - start_mem) / 1024 / 1024)
		peaks["w"].append((torch.cuda.max_memory_allocated() - before_w) / 1024 / 1024)

	median = lambda values: {k: np.median(v).item() for k, v in values.items()}  # noqa: E731
	return (
		median(times),
		median(mems),
		median(peaks),
		np.median(saved_inputs).item(),
		minput,
		moutput,
		warmup_output,
	)


def _measure_remat_forward(stage: torch.nn.Module, sample: tuple[torch.Tensor, ...]) -> float:
	"""
	Peak memory of a forward that keeps nothing, which is how a fully rematerialized stage runs it.

	Measured through the rematerialization engine rather than through something that behaves like
	it, so that what the ILP is given is what the runtime will do.

	:rtype: float
	"""
	# imported here because elf.scheduling imports this module, and elf.execution imports
	# elf.scheduling
	from elf.execution.remat import Stage, unit_placement
	from elf.scheduling import RematMode

	strategy = unit_placement(
		lambda name, module: RematMode.CHECKPOINT if name == "" else None, replayable=False
	)

	torch.cuda.reset_peak_memory_stats()
	start_mem = torch.cuda.memory_allocated()
	output = Stage(stage, strategy, replayable=False).run(*(x.get() for x in _stage_inputs(sample)))
	peak = (torch.cuda.max_memory_allocated() - start_mem) / 1024 / 1024
	del output

	return peak


def _measure_communication(
	signatures: list[Signature],
	placement: list[int],
	in_metas: dict[int, tuple[TensorMetadata, ...]],
	out_metas: dict[int, tuple[TensorMetadata, ...]],
	n_iter: int = 100,
	group=None,
) -> dict[int, float]:
	# TODO: temporary, only the transfer of the first output to its first target is measured
	rank = dist.get_rank()
	times = {block_id: 0.0 for block_id in out_metas}
	torch.cuda.synchronize()
	for source, signature in enumerate(signatures):
		target = signature.targets[0][0]
		if target is None or placement[source] == placement[target]:
			continue

		if placement[source] == rank:
			buffer = out_metas[source][0].get_buffer(1).squeeze(0)
			samples = []
			for _ in range(n_iter):
				with Timer() as timer:
					dist.send(buffer, placement[target], group=group)
				samples.append(timer.time() * 1000)  # in ms
			times[source] = np.median(samples).item()

		elif placement[target] == rank:
			buffer = in_metas[target][signatures[target].sources.index(source)].get_buffer(1).squeeze(0)
			for _ in range(n_iter):
				dist.recv(buffer, placement[source], group=group)

	return times


def profile_stage(
	stage: torch.nn.Module, sample_metadata: TensorMetadata, n_iter: int = 10
) -> tuple[StageProfile, torch.Tensor]:
	"""
	Measure one stage on its own, leaving :py:attr:`StageProfile.t_transfer` at zero.

	The transfer time is left to :func:`profile_pipeline`, since measuring it needs every rank to
	post its side at the same moment.

	:param stage: the module to profile
	:type stage: torch.nn.Module
	:param sample_metadata: descriptor of the stage's inputs, from :func:`propagate_sample`
	:type sample_metadata: TensorMetadata
	:param n_iter: how many times each operation is measured; the median is kept
	:type n_iter: int

	:return: the profile, and the detached output of the warmup pass, ready to be the next
		stage's input
	:rtype: Tuple[StageProfile, torch.Tensor]
	"""
	sample = tuple(meta.get_buffer(1).squeeze(0) for meta in sample_metadata)
	for s in sample:
		init_tensor_(s)

	T, M, Mpeak, saved_inputs, minput, moutput, next_input = _measure_full_pass(stage, sample, n_iter)
	peak_remat = _measure_remat_forward(stage, sample)

	profile = StageProfile(
		t_forward=T["f"],
		t_backward_inputs=T["b"],
		t_backward_params=T["w"],
		t_transfer=0.0,
		m_forward=M["f"],
		m_backward_inputs=M["b"],
		m_backward_params=M["w"],
		m_peak_forward=Mpeak["f"],
		m_peak_backward_inputs=Mpeak["b"],
		m_peak_backward_params=Mpeak["w"],
		m_peak_forward_no_grad=peak_remat,
		m_parameters=sum(p.nbytes for p in stage.parameters()) / 1024 / 1024,
		m_inputs=minput,
		m_outputs=moutput,
		m_saved_inputs=saved_inputs,
	)
	return profile, next_input


def propagate_sample(block_models, signatures, placement, sample, pp_group):
	"""Determine per-block input shapes without running real computation.

	Uses ``FakeTensorMode`` to trace shapes through each block, following the dataflow described by
	``signatures``: an input whose source is ``None`` is taken from ``sample``, in order, and any
	other input is an output of its source block. As in the engine, the outputs of block ``i``
	that go to block ``j`` are matched by order with the inputs of ``j`` that come from ``i``.

	Every rank walks the blocks in id order and, before block ``j``, handles each of its incoming
	edges in source order, so the blocking sends and receives line up across ranks. At rank
	boundaries only a :class:`TensorMetadata` descriptor is exchanged via GPU p2p.

	:param block_models: dict mapping block id -> nn.Module (only for blocks on this rank)
	:param signatures: list of :class:`Signature` for all blocks
	:param placement: stage-to-rank mapping
	:param sample: original model input (on any device)
	:param pp_group: pipeline-parallel process group
	:return: two dicts mapping local block ids to the :class:`TensorMetadata` of each block's
		inputs and of its outputs
	"""
	rank = dist.get_rank()

	block_in_metas = {}
	out_metas = {}  # block id -> one TensorMetadata per output, for blocks on this rank
	fake_mode = FakeTensorMode(
		allow_non_fake_inputs=True
	)  # necessary unless we also convert all parameters
	converter = fake_mode.fake_tensor_converter
	sample = (sample,) if not isinstance(sample, tuple) else sample
	fake_sample = tuple(
		converter.from_real_tensor(fake_mode, s)
		for s in sample  # hope every argument is a tensor
	)

	for block_id, signature in enumerate(signatures):
		owner = placement[block_id]

		in_meta = [None] * signature.ninputs
		fake_inp = [None] * signature.ninputs
		external = [i for i, src in enumerate(signature.sources) if src is None]
		if owner == rank:
			assert len(external) <= len(fake_sample), (
				f"Block {block_id} takes {len(external)} model inputs, the sample only has "
				f"{len(fake_sample)}"
			)
			for i, s in zip(external, fake_sample):
				in_meta[i] = TensorMetadata(s)
				fake_inp[i] = s

		for src in sorted(s for s in signature.get_all_sources() if s is not None):
			src_owner = placement[src]
			if owner != rank and src_owner != rank:
				continue

			inputs = [i for i, s in enumerate(signature.sources) if s == src]
			outputs = [o for o, tgt in enumerate(signatures[src].targets) if block_id in tgt]
			assert len(inputs) == len(outputs), (
				f"Block {block_id} expects {len(inputs)} inputs from block {src}, which sends it "
				f"{len(outputs)} outputs"
			)

			if owner == rank and src_owner == rank:
				for i, o in zip(inputs, outputs):
					in_meta[i] = out_metas[src][o]
			elif src_owner == rank:
				for o in outputs:
					dist.send(out_metas[src][o].to_tensor(), dst=owner, group=pp_group)
			else:
				for i in inputs:
					meta_buf = torch.empty(TensorMetadata.MAX_SIZE, device="cuda")
					dist.recv(meta_buf, src=src_owner, group=pp_group)
					in_meta[i] = TensorMetadata.from_tensor(meta_buf)

		if owner != rank:
			continue

		with fake_mode, torch.no_grad():
			# buffer has an additional dimension for the batch size, but it's already there in the tensor
			fake_inp = tuple(
				x if x is not None else meta.get_buffer(1).squeeze(0) for x, meta in zip(fake_inp, in_meta)
			)
			# TODO: what happens if some arguments are not tensors?

			block_in_metas[block_id] = tuple(in_meta)
			fake_out = block_models[block_id](*fake_inp)
			fake_out = (fake_out,) if not isinstance(fake_out, tuple) else fake_out

			out_metas[block_id] = tuple(TensorMetadata(s) for s in fake_out)

		assert len(out_metas[block_id]) == signature.noutputs, (
			f"Block {block_id} returned {len(out_metas[block_id])} outputs, its signature declares "
			f"{signature.noutputs}"
		)

		for module in block_models[block_id].modules():
			if isinstance(module, LayerDW):
				module.clear()

	return block_in_metas, out_metas


def profile_pipeline(blocks, signatures, placement, sample, pp_group) -> list[StageProfile]:
	"""
	Profile every stage of the pipeline, in parallel across the ranks.

	1. :func:`propagate_sample` determines per-block input shapes.
	2. Each rank profiles its own blocks with :func:`profile_stage`.
	3. The transfer times are measured one link at a time, every rank walking the blocks in the
	   same order, so that the sends and receives across ranks match.
	4. The profiles are gathered across the PP group, so that every rank ends up knowing the whole
	   pipeline -- which is what the schedulers need, since each rank solves for the whole
	   schedule and then keeps its own part.

	:param blocks: list of :class:`~elf.execution.PipelineBlock` on this rank
	:param signatures: list of :class:`Signature` for all blocks
	:param placement: stage-to-rank mapping
	:param sample: original model input
	:param pp_group: pipeline-parallel process group

	:return: one profile per stage, in stage order
	:rtype: List[StageProfile]
	"""
	if dist.get_rank() == 0:
		logger.info("Profiling pipeline stages..")

	block_models = {b.id: b.model for b in blocks}
	block_in_metas, block_out_metas = propagate_sample(
		block_models, signatures, placement, sample, pp_group
	)

	local = {}
	for block in sorted(blocks, key=lambda b: b.id):
		profile, _ = profile_stage(block.model, block_in_metas[block.id])
		block.model.zero_grad(set_to_none=True)
		for module in block.model.modules():
			if isinstance(module, LayerDW):
				module.clear()
		torch.cuda.empty_cache()
		local[block.id] = profile

	transfer_times = _measure_communication(
		signatures, list(placement), block_in_metas, block_out_metas, group=pp_group
	)
	for block_id, profile in local.items():
		profile.t_transfer = transfer_times[block_id]

	gathered = [None] * dist.get_world_size(pp_group)
	dist.all_gather_object(gathered, local, group=pp_group)

	profiles = [None] * len(placement)
	for rank_profiles in gathered:
		if rank_profiles is not None:
			for block_id, profile in rank_profiles.items():
				profiles[block_id] = profile

	assert all(p is not None for p in profiles), "some stages were not profiled"
	return profiles
