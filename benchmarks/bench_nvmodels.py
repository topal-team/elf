import os
import torch
import torch.distributed as dist
import datetime
import sys
import argparse
import wandb

sys.path.append(".")
from elf.zb_utils import replace_layer_with_layer_dw
from models.nvmodels import UNet2D, Autoencoder2D, DiT  # noqa: F401
from elf import Pipeline
from elf.utils import Timer

import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("main")


def get_model_sample_target(model, batch_size):
	match model:
		case "unet":
			block_out_channels = [160, 224, 352, 576, 960]
			block_out_channels = [round(x, 32) for x in block_out_channels]
			model = UNet2D(
				in_channels=1,
				out_channels=2,
				kernel=[3, 5],
				block_out_channels=block_out_channels,
				regular_padding_mode=["replicate", "circular"],
				up_interpolation_mode="linear",
			).cpu()
			shape = (batch_size, 1, 721, 1440)
			sample = (torch.randn(shape), torch.randn((batch_size, 1)), False)
			target = torch.randn((batch_size, 2, 721, 1440))

			if dist.get_rank() == 0:
				logger.info("Running dummy run...")

			with Timer(type="cpu") as timer:
				with torch.no_grad():
					_ = model(sample[0][0:1], sample[1][0:1], dummy_run=True)

			if dist.get_rank() == 0:
				logger.info("Done in %.2fs.", timer.time())

		case "autoencoder":
			block_out_channels = [160, 224, 352, 576, 960]
			block_out_channels = [round(x, 32) for x in block_out_channels]
			model = Autoencoder2D(
				in_channels=2,
				out_channels=2,
				kernel=[3, 5],
				block_out_channels=block_out_channels,
				regular_padding_mode=["replicate", "circular"],
				up_interpolation_mode="linear",
			).cpu()
			shape = (batch_size, 2, 721, 1440)
			sample = (torch.randn(shape), torch.randn((batch_size, 1)), False, False)
			target = torch.randn(shape)

			if dist.get_rank() == 0:
				logger.info("Running dummy run")

			with Timer(type="cpu") as timer:
				with torch.no_grad():
					# use batch size 1 for dummy run for faster inference
					_ = model(sample[0][0:1], sample[1][0:1], return_latent=False, dummy_run=True)

			if dist.get_rank() == 0:
				logger.info("Done in %.2fs.", timer.time())

		case "dit":
			model = DiT(
				seq_size=(45, 90),
				seq_dim=2 * 934,
				hidden_dim=1152,
				regular_padding_mode="constant",
				out_dim=None,
				depth=28,
				num_heads=16,
				mlp_ratio=4.0,
			).cpu()
			shape = (batch_size, 2 * 934, 45, 90)
			sample = (
				model.img_to_seq_pixel(torch.randn(shape, device="cuda")),
				torch.randn((batch_size,), device="cuda"),
			)
			target = model.img_to_seq_pixel(torch.randn(shape, device="cuda"))

	replace_layer_with_layer_dw(model, "cpu")
	return model, sample, target


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--log", choices=["none", "info", "debug"], default="info")
	parser.add_argument("--model", choices=["unet", "autoencoder", "dit"], default="dit")
	parser.add_argument("--scheduler", choices=["1f1b", "zbh1", "zbh2"], default="1f1b")
	parser.add_argument(
		"--partitioner", choices=["naive", "constrained", "metis", "dagP"], default="naive"
	)
	parser.add_argument("--run-id", type=str, default="", help="Run ID")
	args = parser.parse_args()
	match args.log:
		case "none":
			logging.getLogger().setLevel(100)
		case "info":
			logging.getLogger().setLevel(logging.INFO)
		case "debug":
			logging.getLogger().setLevel(logging.DEBUG)
	return args


def main():
	args = parse_args()
	rank = int(os.environ.get("RANK", 0))
	local_rank = int(os.environ.get("LOCAL_RANK", 0))
	torch.cuda.set_device(local_rank)
	dist.init_process_group(
		backend="nccl",
		device_id=torch.device(f"cuda:{local_rank}"),
		timeout=datetime.timedelta(seconds=300),
	)

	batch_size = dist.get_world_size() * 1

	model, sample, target = get_model_sample_target(args.model, batch_size)

	if rank == 0:
		print(f"# of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
		logger.info(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

	pipeline = Pipeline(model, sample, partitioner=args.partitioner, scheduler=args.scheduler)

	if rank == 0:
		logger.info("Pipeline created. Running pipeline...")

	# Warmup
	if rank == 0:
		logger.info("Warming up...")

	for _ in range(3):
		y, loss = pipeline(sample, target, torch.nn.functional.mse_loss)

	dist.barrier()
	if rank == 0:
		logger.info("Warmup complete")

	n = 50
	with Timer("Pipeline") as timer:
		for i in range(n):
			y, loss = pipeline(sample, target, torch.nn.functional.mse_loss)

	if rank == 0:
		# Log config
		config_dict = {"batch_size": batch_size, "run_id": args.run_id}

		wandb.init(
			project="nvmodels",
			entity="topal-inria",
			job_type="model-training",
			config=config_dict,
			mode="offline",
		)

		time = timer.time()
		# Log metrics
		metrics = {
			"world_size": dist.get_world_size(),
			"partitioner": args.partitioner,
			"scheduler": args.scheduler,
			"model": args.model,
			"throughput": (n * dist.get_world_size()) / time,
			"iteration_time": time / n,
		}

		wandb.log(metrics)
		wandb.finish()

	dist.barrier()
	if rank == 0:
		print(f"Rank {rank} - Time taken for {n} iterations: {time}s")

	dist.destroy_process_group()


if __name__ == "__main__":
	main()
