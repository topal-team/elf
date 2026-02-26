import argparse
from dataclasses import dataclass

from fno.benchmark import FNOBlock
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.flop_counter import FlopCounterMode

# Model and data
from neuralop import FNO
from the_well.data import WellDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import elf
from elf.utils import Timer
from elf.zb_utils import replace_layer_with_layer_dw

from benchmarks.benchmark_utils import init_dist

# import logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

PROFILE = False
MODEL_FLOPS = None
GPU_FLOPS = None

COMPLEX_DATA = True
CHANNELS_LAST = False
ALLOW_TF32 = True

DATASET = "turbulent_radiative_layer_3D"
DATASET_DIMS = (256, 128, 128)


@dataclass
class Flops:
	"""Can represent both FLOPs and FLOPS/s, don't get confused :)"""

	fp32: int
	tf32: int

	def __mul__(self, other):
		if isinstance(other, (int, float)):
			return Flops(fp32=self.fp32 * other, tf32=self.tf32 * other)
		elif isinstance(other, Flops):
			return self.fp32 * other.fp32 + self.tf32 * other.tf32
		else:
			raise TypeError(f"Cannot multiply Flops by {type(other)}")

	def __add__(self, other):
		assert isinstance(other, Flops)
		return Flops(fp32=self.fp32 + other.fp32, tf32=self.tf32 + other.tf32)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		"""
		Used for FLOPs / FLOPS/s calculations
		"""
		if isinstance(other, (int, float)):
			return Flops(fp32=self.fp32 / other, tf32=self.tf32 / other)
		elif isinstance(other, Flops):
			return (self.fp32 / other.fp32) + ((self.tf32 / other.tf32) if other.tf32 != 0 else 0)
		else:
			raise TypeError(f"Cannot divide Flops by {type(other)}")

	def __str__(self):
		return f"Flops(fp32={self._format(self.fp32)}, tf32={self._format(self.tf32)})"

	__repr__ = __str__

	@staticmethod
	def _format(flops: int) -> str:
		if flops > 1e12:
			return f"{flops / 1e12:.1f} TFLOPs"
		elif flops > 1e9:
			return f"{flops / 1e9:.1f} GFLOPs"
		elif flops > 1e6:
			return f"{flops / 1e6:.1f} MFLOPs"
		else:
			return f"{flops:.1f} FLOPs"

	def to_dict(self):
		return {"fp32": self.fp32, "tf32": self.tf32}


GPU_SPECS = {
	"H100": Flops(fp32=67 * 1e12, tf32=495 * 1e12),
	"V100": Flops(fp32=15.7 * 1e12, tf32=125 * 1e12),
}  # if you find 969 TFLOPs, it is WITH SPARSITY


def gather_complex(tensor, dst=0, group=None):
	rank = dist.get_rank()
	tensor = torch.view_as_real(tensor)
	gather_list = [torch.empty_like(tensor) for _ in range(group.size())] if rank == dst else None
	dist.gather(tensor, gather_list, dst=dst, group=group)
	if rank == dst:
		return torch.view_as_complex(torch.cat(gather_list, dim=0))
	return None


def once():
	return not dist.is_initialized() or (dist.get_rank() == 0)


def detect_gpu_flops():
	"""Auto-detect GPU and return its FLOPs/s"""
	gpu_name = torch.cuda.get_device_name(0)
	for name, flops in GPU_SPECS.items():
		if name in gpu_name:
			return flops

	raise ValueError(f"Unknown GPU: {gpu_name}")


def _flops_from_counter(flop_counter):
	"""
	Extract TF32 and FP32 FLOPs from a FlopCounterMode
	"""
	TF32_OPS = (
		torch.ops.aten.convolution,
		torch.ops.aten.convolution_backward,
		torch.ops.aten.addmm,
		torch.ops.aten.bmm,
		torch.ops.aten.baddbmm,
	)
	tf32_enabled = torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32

	tf32_flops = sum(
		flops
		for op, flops in flop_counter.flop_counts["Global"].items()
		if op in TF32_OPS and tf32_enabled
	)
	fp32_flops = flop_counter.get_total_flops() - tf32_flops
	return Flops(fp32=fp32_flops, tf32=tf32_flops)


def compute_model_flops(model, data_shape):
	"""Compute model FLOPs per batch using meta device"""
	with torch.device("meta"):
		meta_model = FNO(
			n_modes=model.n_modes,
			in_channels=model.in_channels,
			out_channels=model.out_channels,
			hidden_channels=model.hidden_channels,
			n_layers=model.n_layers,
			projection_channel_ratio=model.projection_channel_ratio,
			complex_data=model.complex_data,
		)
		dtype = torch.complex64 if model.complex_data else torch.float32
		sample = torch.randn(data_shape, device="meta", dtype=dtype)
		target = torch.randn(data_shape, device="meta", dtype=dtype)

		flop_counter = FlopCounterMode(display=False)
		with flop_counter:
			output = meta_model(sample)
			loss = torch.nn.functional.l1_loss(output, target)
			loss.backward()

	return _flops_from_counter(flop_counter)


def compute_mfu(throughput, ngpus):
	"""Compute Model FLOPs Utilization (MFU). Throughput is in batches/s, model FLOPs are in FLOPs/batch."""
	if MODEL_FLOPS is None or GPU_FLOPS is None:
		return None
	theoretical_throughput = (ngpus * GPU_FLOPS) / MODEL_FLOPS
	return throughput / theoretical_throughput


def preprocess_data(batch):
	inputs, targets = (batch["input_fields"], batch["output_fields"])  # ignore boundaries etc for now
	with torch.no_grad():
		inputs = inputs.to("cuda", non_blocking=True)
		targets = targets.to("cuda", non_blocking=True)
		if COMPLEX_DATA:
			inputs = torch.complex(inputs[..., 0], inputs[..., 1])
			targets = torch.complex(targets[..., 0], targets[..., 1])
		else:
			inputs = inputs[..., 0]  # real part only
			targets = targets[..., 0]
		if CHANNELS_LAST:
			inputs = inputs.permute(0, 2, 3, 1).contiguous()
			targets = targets.permute(0, 2, 3, 1).contiguous()

		return inputs, targets


def compute_vrmse(predictions, targets, eps=1e-10):
	"""
	Compute VRMSE (Variance-scaled Root Mean Square Error).
	VRMSE is scaled such that predicting the mean value of the target field results in a score of 1.

	VRMSE(u,v) = sqrt(⟨|u-v|²⟩ / (⟨|u-ū|²⟩ + ϵ))
	where u is ground truth, v is prediction, ū is mean of ground truth.

	Args:
		predictions: Model predictions (can be complex)
		targets: Ground truth target values (can be complex)
		eps: Small constant for numerical stability

	Returns:
		VRMSE score (lower is better, 1.0 means as good as predicting the mean)
	"""
	with torch.no_grad():
		mse = torch.mean(torch.abs(predictions - targets) ** 2)
		target_mean = torch.mean(targets)
		variance = torch.mean(torch.abs(targets - target_mean) ** 2)
		vrmse = torch.sqrt(mse / (variance + eps))
		return vrmse.real.item()


class EmptyData(torch.utils.data.Dataset):
	def __init__(self, num_samples):
		self.num_samples = num_samples

	def __getitem__(self, index):
		if index >= self.num_samples:
			raise IndexError("index out of range")
		return {
			"input_fields": torch.empty((1, 1024, 256, 2), dtype=torch.float32),
			"output_fields": torch.empty((1, 1024, 256, 2), dtype=torch.float32),
		}

	def __len__(self):
		return self.num_samples


class DataPrefetcher:
	"""Wraps a DataLoader to prefetch and preprocess the next batch on a side CUDA stream,
	overlapping host-to-device transfer with GPU computation."""

	def __init__(self, loader):
		self.loader = loader
		self.stream = torch.cuda.Stream()
		self._iterator = None
		self._next_data = None

	def __iter__(self):
		self._iterator = iter(self.loader)
		self._prefetch()
		return self

	def _prefetch(self):
		try:
			batch = next(self._iterator)
		except StopIteration:
			self._next_data = None
			return
		with torch.cuda.stream(self.stream):
			self._next_data = preprocess_data(batch)

	def __next__(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		data = self._next_data
		if data is None:
			raise StopIteration
		inputs, targets = data
		inputs.record_stream(torch.cuda.current_stream())
		targets.record_stream(torch.cuda.current_stream())
		self._prefetch()
		return inputs, targets

	def __len__(self):
		return len(self.loader)


def distribute_data(dataset, group_ranks, batch_size, num_workers=1, persistent_workers=False):
	"""
	Distribute data such that ranks in the same group get identical data.
	group_ranks: list of rank groups, e.g., [[0,1], [2,3]]
	"""

	rank = dist.get_rank()
	num_groups = len(group_ranks)
	for group_idx, group in enumerate(group_ranks):
		if rank in group:
			torch.manual_seed(group_idx)
			sampler = DistributedSampler(dataset, num_replicas=num_groups, rank=group_idx, shuffle=True)
			return DataLoader(
				dataset,
				batch_size=batch_size // num_groups,
				sampler=sampler,
				num_workers=num_workers,
				pin_memory=True,
				persistent_workers=persistent_workers and num_workers > 0,
			)


def _get_current_fragmentation(device=None):
	"""
	Computes a fragmentation score for the current GPU memory state.

	Uses the L2/L1 norm-based metric from
	https://asawicki.info/news_1757_a_metric_for_memory_fragmentation:

	.. math::
		F = 1 - \\left(\\frac{\\sqrt{\\sum f_i^2}}{\\sum f_i}\\right)^2
		  = 1 - \\frac{\\sum f_i^2}{\\left(\\sum f_i\\right)^2}

	where :math:`f_i` are the sizes of contiguous free regions. Unlike the
	classical ``1 - largest_free / total_free``, this accounts for the full
	distribution of free regions: small holes barely affect the score when
	large regions exist, while uniformly scattered free memory pushes it
	toward 1.

	Free memory includes both inactive (cached but unused) blocks inside
	PyTorch's caching allocator segments and unclaimed CUDA memory that
	PyTorch has not yet reserved. Consecutive inactive blocks within a
	segment are merged, since they form contiguous free regions.

	:param device: CUDA device index. Defaults to the current device.
	:type device: int or None

	:return: Fragmentation score in [0, 1]. 0 means all free memory is
		contiguous, values approaching 1 indicate severe fragmentation.
	:rtype: float
	"""
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available")

	if device is None:
		device = torch.cuda.current_device()

	torch.cuda.synchronize(device)
	snapshot = torch.cuda.memory_snapshot()

	# Merge consecutive inactive blocks within each segment to get
	# the true contiguous free region sizes.
	free_regions = []
	for segment in snapshot:
		if segment.get("device", 0) != device:
			continue
		current_free_run = 0
		for block in segment["blocks"]:
			if block["state"] == "inactive":
				current_free_run += block["size"]
			else:
				if current_free_run > 0:
					free_regions.append(current_free_run)
				current_free_run = 0
		if current_free_run > 0:
			free_regions.append(current_free_run)

	# Unclaimed CUDA memory (not yet reserved by the caching allocator)
	# is available via cudaMalloc and counts as one free region.
	cuda_free, _ = torch.cuda.mem_get_info(device)
	if cuda_free > 0:
		free_regions.append(cuda_free)

	total_free = sum(free_regions)
	if total_free == 0:
		return 0.0

	sum_of_squares = sum(f * f for f in free_regions)
	return 1.0 - sum_of_squares / (total_free * total_free)


def train_base(data, model, epochs, batch_size):
	model.cuda()
	model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	train_loader = DataLoader(
		data,
		batch_size=batch_size,
		shuffle=True,
		pin_memory=True,
		num_workers=4,
		persistent_workers=True,
	)

	if PROFILE:
		for name, module in model.named_modules():
			module._name = name

			def nvtx_pre_hook(module, *args):
				torch.cuda.nvtx.range_push(f"{module._name}")

			def nvtx_post_hook(module, *args):
				torch.cuda.nvtx.range_pop()

			module.register_forward_pre_hook(nvtx_pre_hook)
			module.register_forward_hook(nvtx_post_hook)

	prefetcher = DataPrefetcher(train_loader)

	for epoch in range(epochs):
		torch.cuda.reset_peak_memory_stats()
		epoch_loss = 0
		with Timer() as timer:
			for inputs, targets in prefetcher:
				optimizer.zero_grad()
				outputs = model(inputs)
				loss = torch.nn.functional.l1_loss(outputs, targets)
				epoch_loss += loss.detach()
				loss.backward()
				optimizer.step()

		if once():
			throughput = len(data) / timer.time()
			print(f"Epoch {epoch} time: {timer.time():.2f}s")
			print(f"Throughput: {throughput:.2f} samples/s")
			mfu = compute_mfu(throughput / batch_size, 1)
			if mfu is not None:
				print(f"MFU: {100 * mfu:.2f}%")
			print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB")
			print(f"Loss: {epoch_loss.item() / len(train_loader):.4f}\n")
			print(f"Current fragmentation: {_get_current_fragmentation():.2f} (lower is better in [0,1])")

	return model


def train_elf(data, model, epochs, batch_size):
	world_size = dist.get_world_size()
	scheduler = "zbh2"
	replace_layer_with_layer_dw(model)
	# model.cuda()

	dp = 1
	pp = world_size // dp
	groups = [list(range(pp * i, pp * (i + 1))) for i in range(dp)]

	# Manual partitioning
	rank = dist.get_rank()
	placement = elf.Placement.default(scheduler, pp)
	parts = [
		partition(model, i, len(placement))
		for i, k in enumerate(placement)
		if (rank % len(placement)) == k
	]
	pipe = elf.Pipeline(parts, partitioner=False, scheduler=scheduler, dp=dp)

	# Only head (inputs) and tail (targets) ranks need real data; middle ranks skip disk I/O
	is_head = pipe.placement.is_head(rank)
	is_tail = pipe.placement.is_tail(rank)
	if is_head or is_tail:
		loader = distribute_data(data, groups, batch_size, num_workers=4, persistent_workers=True)
	else:
		loader = distribute_data(EmptyData(len(data)), groups, batch_size)

	# Auto partitioning
	# sample = torch.randn((1, 1, 1024, 256), device="cuda", dtype=torch.complex64)
	# pipe = elf.Pipeline(model, sample=sample, partitioner="dagP", scheduler=scheduler, dp=dp)

	optimizer = torch.optim.Adam(pipe.parameters(), lr=0.001)

	prefetcher = DataPrefetcher(loader)

	for epoch in range(epochs):
		epoch_losses = []
		with Timer() as timer:
			torch.cuda.reset_peak_memory_stats()
			for inputs, targets in prefetcher:
				optimizer.zero_grad()

				y, loss = pipe(
					inputs,
					targets if is_tail else None,
					loss_fn=torch.nn.functional.l1_loss,
					split_size=1,
					profile=PROFILE,
				)
				if loss is not None:
					epoch_losses.append(loss.detach())

				optimizer.step()

			dist.barrier()

		if once():
			throughput = len(data) / timer.time()
			print(f"Epoch {epoch} time: {timer.time():.2f}s")
			print(f"Throughput: {throughput:.2f} samples/s")
			mfu = compute_mfu(throughput / batch_size, world_size)
			if mfu is not None:
				print(f"MFU: {100 * mfu:.2f}%")
			print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB")

		if epoch_losses and pipe.dp_rank == 0:
			loss = torch.mean(torch.tensor(epoch_losses, device="cuda"))
			print(f"Loss: {loss.item():.4f}\n")

	return pipe


def train_ddp(data, model, epochs, batch_size):
	if not dist.is_initialized():
		raise RuntimeError("Distributed process group must be initialized before calling train_ddp")

	world_size = dist.get_world_size()

	groups = [[i] for i in range(world_size)]
	train_loader = distribute_data(data, groups, batch_size, num_workers=4, persistent_workers=True)

	model.cuda()
	model = DDP(model, device_ids=[torch.cuda.current_device()])
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	prefetcher = DataPrefetcher(train_loader)

	for epoch in range(epochs):
		torch.cuda.reset_peak_memory_stats()
		epoch_loss = torch.tensor([0.0], device=torch.cuda.current_device())
		with Timer() as timer:
			for inputs, targets in prefetcher:
				optimizer.zero_grad()
				outputs = model(inputs)
				loss = torch.nn.functional.l1_loss(outputs, targets)
				epoch_loss += loss.detach()
				loss.backward()
				optimizer.step()

		dist.reduce(epoch_loss, dst=0, op=dist.ReduceOp.AVG)
		if once():
			throughput = len(data) / timer.time()
			print(f"Epoch {epoch} time: {timer.time():.2f}s")
			print(f"Throughput: {throughput:.2f} samples/s")
			mfu = compute_mfu(throughput / batch_size, world_size)
			if mfu is not None:
				print(f"MFU: {100 * mfu:.2f}%")
			print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB")
			print(f"Loss: {epoch_loss.item() / len(train_loader):.4f}\n")

	return model


def evaluate_base(model, test_dataset):
	model.eval()
	all_predictions = []
	all_targets = []
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
	with torch.no_grad():
		for batch in test_loader:
			inputs, targets = preprocess_data(batch)
			outputs = model(inputs)
			all_predictions.append(outputs)
			all_targets.append(targets)

	vrmse = compute_vrmse(torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0))
	print(f"\nTest VRMSE: {vrmse:.4f}")


def evaluate_elf(pipe, test_dataset):
	rank = dist.get_rank()

	groups = [list(range(pipe.pp * i, pipe.pp * (i + 1))) for i in range(pipe.dp)]
	test_loader = distribute_data(test_dataset, groups, len(groups))

	first_tail = pipe.placement.tail() - (pipe.dp_rank * pipe.pp)

	local_predictions = []
	local_targets = []

	with torch.no_grad():
		for batch in test_loader:
			inputs, targets = preprocess_data(batch)
			outputs, *_ = pipe(inputs, None, None, split_size=1, scheduler="inference")

			if pipe.placement.is_tail(rank):
				local_predictions.extend(outputs)
				local_targets.append(targets)

		if pipe.placement.is_tail(rank):
			all_predictions = torch.cat(local_predictions, dim=0)
			all_targets = torch.cat(local_targets, dim=0)
			if pipe.dp > 1:
				all_predictions = gather_complex(
					all_predictions, dst=first_tail, group=pipe.blocks[-1].dp_group
				)
				all_targets = gather_complex(all_targets, dst=first_tail, group=pipe.blocks[-1].dp_group)

		if rank == first_tail:
			print(f"\nTest VRMSE: {compute_vrmse(all_predictions, all_targets):.4f}")


def evaluate_ddp(model, test_dataset):
	rank = dist.get_rank()
	world_size = dist.get_world_size()

	groups = [[i] for i in range(world_size)]
	test_loader = distribute_data(test_dataset, groups, len(groups))
	model.eval()

	local_predictions = []
	local_targets = []

	with torch.no_grad():
		for batch in test_loader:
			inputs, targets = preprocess_data(batch)
			outputs = model(inputs)
			local_predictions.append(outputs)
			local_targets.append(targets)

		local_predictions = torch.view_as_real(
			torch.cat(local_predictions, dim=0)
		)  # NCCL cannot handle complex tensors
		local_targets = torch.view_as_real(torch.cat(local_targets, dim=0))
		all_predictions = (
			[torch.empty_like(local_predictions) for _ in range(world_size)] if rank == 0 else None
		)
		all_targets = (
			[torch.empty_like(local_targets) for _ in range(world_size)] if rank == 0 else None
		)
		dist.gather(local_predictions, all_predictions, dst=0)
		dist.gather(local_targets, all_targets, dst=0)
		if rank == 0:
			all_predictions = torch.view_as_complex(torch.cat(all_predictions, dim=0))
			all_targets = torch.view_as_complex(torch.cat(all_targets, dim=0))

			print(f"\nTest VRMSE: {compute_vrmse(all_predictions, all_targets):.4f}")


def evaluate(model_or_pipe, test_dataset, mode):
	"""
	Evaluate model on test set and compute VRMSE metric.
	Works with base, elf, and ddp modes.
	"""
	match mode:
		case "base":
			evaluate_base(model_or_pipe, test_dataset)
		case "elf":
			evaluate_elf(model_or_pipe, test_dataset)
		case "ddp":
			evaluate_ddp(model_or_pipe, test_dataset)
		case _:
			raise ValueError(f"Invalid mode: {mode}")


def partition(model, rank, ws):
	part = nn.Sequential()
	if rank == 0:
		part.append(model.positional_embedding)
		part.append(model.lifting)

	n_blocks = model.fno_blocks.n_layers
	base = n_blocks // ws
	remainder = n_blocks % ws
	balanced = [base] * ws
	for i in range(2, ws + 1):
		if remainder == 0:
			break
		balanced[-i] += 1
		remainder -= 1

	start = sum(balanced[:rank])
	for i in range(balanced[rank]):
		part.append(FNOBlock(model.fno_blocks, start + i))

	if rank == ws - 1:
		part.append(model.projection)

	return part.cuda()


def save(model, path, mode):
	if mode == "base" or mode == "ddp":
		torch.save(model.state_dict(), path)
	elif mode == "elf":
		state_dict = model.gather_parameters(dst=0)
		if dist.get_rank() == 0:
			torch.save(state_dict, path)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type=str, required=True, help="Path to the data")
	parser.add_argument("--mode", type=str, choices=["base", "elf", "ddp"], default="base")
	parser.add_argument("--profile", action="store_true", help="Profile the training")
	parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
	parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
	parser.add_argument("--save", type=str, default=None, help="Path to save the model")
	parser.add_argument("--no-tf32", action="store_true", help="Disable TF32")
	parser.add_argument("--no-complex", action="store_true", help="Disable complex data")
	parser.add_argument("--compile", action="store_true", help="Compile the model")
	return parser.parse_args()


def main():
	global PROFILE, MODEL_FLOPS, GPU_FLOPS, COMPLEX_DATA, CHANNELS_LAST, ALLOW_TF32

	args = parse_args()
	if args.mode in ("elf", "ddp"):
		init_dist()

	if args.no_tf32:
		ALLOW_TF32 = False

	torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
	torch.backends.cudnn.allow_tf32 = ALLOW_TF32

	if args.no_complex:
		COMPLEX_DATA = False

	data = WellDataset(
		well_base_path=args.data,
		well_dataset_name=DATASET,
		well_split_name="test",
		restrict_num_samples=64,
		boundary_return_type=None,
	)
	if once():
		print(f"Training dataset has {len(data)} samples")

	model = FNO(
		n_modes=(16, 16, 16),
		in_channels=1,
		out_channels=1,
		hidden_channels=64,
		n_layers=4,
		complex_data=not args.no_complex,
	)
	if args.compile:
		model = torch.compile(model, mode="max-autotune-no-cudagraphs")

	if once():
		print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

	MODEL_FLOPS = compute_model_flops(model, (args.batch_size, 1, *DATASET_DIMS))
	GPU_FLOPS = detect_gpu_flops()
	if once():
		print(f"Model FLOPs: {MODEL_FLOPS}/sample")
		print(f"GPU FLOPs: {GPU_FLOPS}/s")
		ngpus = dist.get_world_size() if dist.is_initialized() else 1
		print(
			f"Theoretical throughput: {args.batch_size * ngpus * GPU_FLOPS / MODEL_FLOPS:.2f} samples/s"
		)
		print()

	if args.profile:
		PROFILE = True
		torch.cuda.synchronize()
		torch.cuda.cudart().cudaProfilerStart()

	match args.mode:
		case "base":
			trained_model = train_base(data, model, args.epochs, args.batch_size)
		case "elf":
			trained_model = train_elf(data, model, args.epochs, args.batch_size)
		case "ddp":
			trained_model = train_ddp(data, model, args.epochs, args.batch_size)
		case _:
			raise ValueError(f"Invalid mode: {args.mode}")

	if args.profile:
		torch.cuda.synchronize()
		torch.cuda.cudart().cudaProfilerStop()

	test_data = WellDataset(
		well_base_path=args.data,
		well_dataset_name=DATASET,
		well_split_name="test",
		restrict_num_samples=32,
		boundary_return_type=None,
	)
	if once():
		print(f"Test dataset has {len(test_data)} samples")

	if once():
		print("\n" + "=" * 50)
		print("Evaluating on test set...")
		print("=" * 50)

	# evaluate(trained_model, test_data, args.mode)

	if args.save:
		save(trained_model, args.save, args.mode)

	if args.mode in ("elf", "ddp") and dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
