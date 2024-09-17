import torch
import torch.nn as nn
import torch.nn.functional as F


class ReversibleColumnBlock(nn.Module):
	def __init__(self, channels, kernel_size=3, padding=1):
		super(ReversibleColumnBlock, self).__init__()
		self.conv1 = nn.Conv2d(channels // 2, channels // 2, kernel_size, padding=padding)
		self.conv2 = nn.Conv2d(channels // 2, channels // 2, kernel_size, padding=padding)
		self.ln1 = nn.LayerNorm([channels // 2, 224, 224])  # Adjusted for 224x224 input
		self.ln2 = nn.LayerNorm([channels // 2, 224, 224])  # Adjusted for 224x224 input

	def forward(self, x):
		x1, x2 = torch.chunk(x, 2, dim=1)
		y1 = x1 + self.conv1(F.relu(self.ln1(x2)))
		y2 = x2 + self.conv2(F.relu(self.ln2(y1)))
		return torch.cat([y1, y2], dim=1)

	def inverse(self, y):
		y1, y2 = torch.chunk(y, 2, dim=1)
		x2 = y2 - self.conv2(F.relu(self.ln2(y1)))
		x1 = y1 - self.conv1(F.relu(self.ln1(x2)))
		return torch.cat([x1, x2], dim=1)


class ReversibleColumnNetwork(nn.Module):
	def __init__(self, in_channels, out_channels, num_blocks, kernel_size=3, padding=1):
		super(ReversibleColumnNetwork, self).__init__()
		self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
		self.blocks = nn.ModuleList(
			[ReversibleColumnBlock(out_channels, kernel_size, padding) for _ in range(num_blocks)]
		)
		self.out_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1)

	def forward(self, x):
		x = self.in_conv(x)
		for block in self.blocks:
			x = block(x)
		x = self.out_conv(x)
		return x

	def inverse(self, y):
		y = self.out_conv.weight.transpose(0, 1).unsqueeze(-1).unsqueeze(-1) * y
		for block in reversed(self.blocks):
			y = block.inverse(y)
		y = self.in_conv.weight.transpose(0, 1).unsqueeze(-1).unsqueeze(-1) * y
		return y


class RCN(nn.Module):
	def __init__(self, in_channels, hidden_channels, num_blocks, num_columns):
		super(RCN, self).__init__()
		self.columns = nn.ModuleList(
			[
				ReversibleColumnNetwork(in_channels, hidden_channels, num_blocks)
				for _ in range(num_columns)
			]
		)

		# Add global average pooling and classifier for CIFAR-100
		self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.classifier = nn.Linear(in_channels, 100)  # 100 classes for CIFAR-100

	def forward(self, x):
		for column in self.columns:
			x = column(x)

		# Apply global average pooling
		x = self.global_avg_pool(x)
		x = x.view(x.size(0), -1)

		# Apply classifier
		x = self.classifier(x)
		return x

	def inverse(self, y):
		# Note: The inverse function is not applicable when using the classifier
		raise NotImplementedError("Inverse operation is not supported with the classifier")
