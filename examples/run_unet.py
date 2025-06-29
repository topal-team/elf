import torch
import torch.nn as nn
import torch.distributed as dist
import os

import sys

sys.path.append(".")
from elf import Pipeline

import logging

logging.basicConfig(level=logging.DEBUG)


class EncoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(EncoderBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool2d(2)

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.pool(x)
		return x


class DecoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(DecoderBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.relu = nn.ReLU(inplace=True)
		self.upsample = nn.Upsample(scale_factor=2)

	def forward(self, x, residual):
		x = self.relu(self.conv1(x))
		x = self.upsample(x)
		return x + residual


class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()

		self.conv1 = EncoderBlock(1, 64)
		self.conv2 = EncoderBlock(64, 128)

		self.conv3 = DecoderBlock(128, 64)
		self.conv4 = DecoderBlock(64, 1)

	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2, x1)
		x4 = self.conv4(x3, x)
		return x4


def main():
	local_rank = int(os.environ.get("LOCAL_RANK", 0))
	torch.cuda.set_device(local_rank)
	dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
	rank = dist.get_rank()

	model = UNet()
	img = torch.randn(4, 1, 256, 256)
	if rank == 0:
		trace = torch.fx.symbolic_trace(model)
		print(trace.code)

	pipeline = Pipeline(model, img, partitioner="dagP", scheduler="1f1b")

	batch = torch.randn(16, 1, 256, 256)
	target = torch.randn(16, 1, 256, 256)
	loss_fn = nn.functional.mse_loss
	_ = pipeline(batch, target, loss_fn)

	print(f"Rank {rank} - Finished.")

	dist.destroy_process_group()


if __name__ == "__main__":
	main()
