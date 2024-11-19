import os
import sys
import copy
import torch
from torch.optim import SGD
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append("./")
from pipeline.pipeline import Pipeline
from models.simple import SimpleTransformer, SimpleCNN, SimpleResNet

from argparse import ArgumentParser
import logging

logger = logging.getLogger("main")
logging.basicConfig(level=logging.INFO)

absolute_tolerance = 1e-6
relative_tolerance = 1e-4


def assert_model_params_equal(model, pg):
	"""
	Assert that all parameters of the model are the same on the two specified ranks.
	(used to check across DP parallelism)

	Args:
		model (torch.nn.Module): The model whose parameters are to be compared.
		rank (int): The rank of the current process.
		world_size (int): The total number of processes.
	"""
	rank = dist.get_rank(pg)
	world_size = dist.get_world_size(pg)
	equals = True
	ranks = [dist.get_global_rank(pg, i) for i in range(0, world_size)]
	dst = dist.get_global_rank(pg, 0)

	for name, param in sorted(model.named_parameters()):
		tensor = param.data.clone()

		# Gather tensors from all ranks
		gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)] if rank == 0 else None
		dist.gather(tensor, gathered_tensors, dst=dst, group=pg)

		# Compare tensors
		if rank == 0:
			for i in range(1, world_size):
				if not torch.equal(gathered_tensors[0], gathered_tensors[i]):
					print(f"Parameter {name} differ between rank {ranks[0]} and rank {ranks[i]}")
					equals = False

	if equals:
		print(f"Model parameters DP check passed successfully for ranks {ranks}")


def merge_dicts(list_of_dicts):
	merged_dict = {}
	for d in list_of_dicts:
		merged_dict.update(d)

	return merged_dict


def get_all_parameters(model, pg):
	params = [{} for _ in range(dist.get_world_size(group=pg))] if dist.get_rank() == 0 else None
	local_params = {name: p.data for name, p in model.named_parameters()}
	dist.gather_object(local_params, params, dst=0, group=pg)

	grads = [{} for _ in range(dist.get_world_size(group=pg))] if dist.get_rank() == 0 else None
	local_grads = {
		name: p.grad.data
		if p.grad is not None
		else torch.tensor([0.0], device=torch.cuda.current_device())
		for name, p in model.named_parameters()
	}
	dist.gather_object(local_grads, grads, dst=0, group=pg)
	if rank == 0:
		return merge_dicts(params), merge_dicts(grads)
	else:
		return None, None


def check_model_parameters(model1, params_model2, grads_model2):
	# Function to check if two tensors are equal
	def tensors_are_equal(tensor1, tensor2):
		return torch.allclose(tensor1, tensor2, rtol=relative_tolerance, atol=absolute_tolerance)

	# Function to check parameters and gradients
	def check_params_and_grads(param_dict1, param_dict2, is_grad):
		equals = True
		for key in param_dict1.keys():
			if key in param_dict2 and not tensors_are_equal(param_dict1[key], param_dict2[key]):
				name = "gradient of" if is_grad else "parameter"
				print(
					f"Mismatch found in {name} {key}: (diff = {torch.linalg.norm(param_dict1[key] - param_dict2[key])})"
				)
				equals = False

		return equals

	# Extract parameters and gradients from the model on GPU 0
	params_model1 = {name: param.data.to(0) for name, param in model1.named_parameters()}
	grads_model1 = {
		name: param.grad.data.to(0) if param.grad is not None else torch.tensor([0.0], device=0)
		for name, param in model1.named_parameters()
	}

	params_model2 = {name: param.to(0) for name, param in params_model2.items()}
	grads_model2 = {name: grad.to(0) for name, grad in grads_model2.items()}

	# Check parameters and gradients
	params_equal = check_params_and_grads(params_model1, params_model2, is_grad=False)
	grads_equal = check_params_and_grads(grads_model1, grads_model2, is_grad=True)

	if params_equal and grads_equal:
		print("All parameters and gradients match.")
	else:
		print("Parameters or gradients do not match.")

	return params_equal and grads_equal


class Dummy(Dataset):
	def __init__(self, shape_in, dtype_in, shape_out, dtype_out, n=16) -> None:
		super().__init__()
		self.n = n
		self.data = [
			torch.zeros(shape_in, device=torch.cuda.current_device(), dtype=dtype_in) + i
			for i in range(n)
		]
		self.targets = [
			torch.ones(shape_out, device=torch.cuda.current_device(), dtype=dtype_out) for _ in range(n)
		]

	def __getitem__(self, index):
		assert index < self.n
		return self.data[index].clone().detach(), self.targets[index].clone().detach()

	def __len__(self):
		return self.n


def train(pipe, model, data, pp, dp):
	if rank == 0:
		print("Training with PP+DP")
		print("Setup :")
		for d in range(dp):
			print(f"\tDP Rank {d} - pipe -> {[i + (pp * d) for i in range(pp)]}")

	if rank == 0:
		optimizer = SGD(model.parameters())
		loader = DataLoader(data, batch_size=16, shuffle=False)

	optimizer_distributed = SGD(pipe.parameters())
	loader_distributed = DataLoader(
		data,
		batch_size=16 // dp,
		shuffle=False,
		sampler=DistributedSampler(data, num_replicas=dp, rank=rank // pp, shuffle=False),
	)

	loss_fn = model.loss_fn
	if rank < pp:
		for block in pipe.blocks:
			pipe_params, pipe_grads = get_all_parameters(block.model, block.pp_group)
			if rank == 0:
				print("Checking pipeline parameters before training")
				check_model_parameters(model, pipe_params, pipe_grads)

	if rank == 0:
		losses = []
		for e in range(10):
			epoch_loss = torch.tensor([0.0], device=torch.cuda.current_device())
			for x, t in loader:
				optimizer.zero_grad()
				y = model(x)
				loss = loss_fn(y, t)
				loss.backward()
				optimizer.step()
				epoch_loss += loss.detach()
			losses.append(epoch_loss)

	for e in range(10):
		epoch_loss = torch.tensor([0.0], device=torch.cuda.current_device())

		for x, t in loader_distributed:
			optimizer_distributed.zero_grad()
			y, loss_distributed = pipe(x, t, loss_fn=loss_fn)
			if loss_distributed:
				epoch_loss += loss_distributed.detach()
			optimizer_distributed.step()

		if rank == pipe.placement[-1] and rank != 0:
			dist.send(epoch_loss, 0)

		if rank == 0:
			if rank != pipe.placement[-1]:
				dist.recv(epoch_loss, pipe.placement[-1])
			assert torch.allclose(
				epoch_loss, losses[e], rtol=relative_tolerance, atol=absolute_tolerance
			), f"Different losses for epoch {e} - expected {losses[e].item()}, got {epoch_loss.item()}"
			print(f"Loss check passed for epoch {e}")

		for block in pipe.blocks:
			if dp > 1:
				assert_model_params_equal(block.model, block.dp_group)  # check same parameters across DP


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-dp", type=int, required=False, default=2)
	parser.add_argument("-pp", type=int, required=False, default=4)
	parser.add_argument(
		"--log", choices=["debug", "info", "none"], default="info", required=False, help="logging level"
	)
	parser.add_argument(
		"--model",
		"-m",
		choices=["cnn", "tf", "resnet"],
		default="tf",
		required=False,
		help="model to use",
	)
	parser.add_argument(
		"--schedule", "-s", choices=["afab", "1f1b", "hanayo"], default="1f1b", required=False
	)
	parser.add_argument(
		"--interleaving", "-i", type=int, required=False, default=1, help="interleaving degree"
	)
	args = parser.parse_args()
	match args.log:
		case "debug":
			logging.getLogger().setLevel(logging.DEBUG)
		case "info":
			logging.getLogger().setLevel(logging.INFO)
		case "none":
			logging.getLogger().setLevel(100)

	rank = int(os.getenv("RANK"))
	local_rank = int(os.getenv("LOCAL_RANK"))
	dist.init_process_group(backend="nccl")
	torch.cuda.set_device(local_rank)

	match args.model:
		case "cnn":
			model = SimpleCNN(256)
			data = Dummy((3, 224, 224), torch.float32, (), torch.int64)
		case "tf":
			model = SimpleTransformer(128, 128, args.pp * args.interleaving * 2)
			data = Dummy((64,), torch.int64, (64,), torch.int64)
		case "resnet":
			model = SimpleResNet(args.pp * args.interleaving)
			data = Dummy((3, 224, 224), torch.float32, (), torch.int64)

	model = model.cuda()

	match args.schedule:
		case "afab" | "1f1b":
			placement = [i for i in range(args.pp)] * args.interleaving
		case "hanayo":
			placement = [i for i in range(args.pp)] + [i for i in reversed(range(args.pp))]
			placement *= args.interleaving
		case _:
			raise ValueError(f"Unknown schedule {args.schedule}")

	pipe = Pipeline(
		copy.deepcopy(model),
		model.get_sample(4).cuda(),
		placement=placement,
		schedule=args.schedule,
		dp=args.dp,
	)
	gt = copy.deepcopy(model)

	train(pipe, gt, data, args.pp, args.dp)

	sample = model.get_sample(32).cuda()
	target = model.get_target(32).cuda()
	if rank == 0 and sample.is_floating_point():
		print(f"Sample given to pipe : mean = {sample.mean()}, std = {sample.std()}")
	y, _ = pipe(sample.clone(), target.clone(), loss_fn=model.loss_fn)
	pipe.zero_grad(set_to_none=False)
	gt.zero_grad(set_to_none=False)

	block = pipe.blocks[0]
	if rank < args.pp:
		pipe_params, pipe_grads = get_all_parameters(block.model, block.pp_group)

	if rank == 0:
		if sample.is_floating_point():
			print(f"Sample given to single gpu : mean = {sample.mean()}, std = {sample.std()}")

		check_model_parameters(gt, pipe_params, pipe_grads)
		z = gt(sample.clone())

		if pipe.placement[-1] != 0:
			y = torch.empty_like(z)
			dist.recv(y, pipe.placement[-1])

		assert torch.allclose(
			y, z, rtol=relative_tolerance, atol=absolute_tolerance
		), f"Wrong output for pipelined model. Difference norm = {torch.linalg.norm(y - z)}"
		print("Test passed successfully")

	elif rank == pipe.placement[-1]:
		dist.send(y, 0)

	pipe.clear()
