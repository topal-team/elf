import torch
import torch.nn as nn


class SimpleResNet(nn.Module):
	def __init__(self, nblocks=12, channels=64, num_classes=10):
		super(SimpleResNet, self).__init__()
		self.channels = channels

		# Initial convolution
		self.conv1 = nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		# Residual blocks
		self.blocks = []
		for i in range(nblocks):
			block = self._make_layer(channels, channels)
			self.add_module(f"block_{i}", block)
			self.blocks.append(block)

		# Final layers
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(channels, num_classes)

	def _make_layer(self, in_channels, out_channels, stride=1):
		layers = []

		# First block handles dimension changes
		layers.append(ResBlock(in_channels, out_channels, stride))

		# Remaining blocks
		for _ in range(1, 4):
			layers.append(ResBlock(out_channels, out_channels))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		for block in self.blocks:
			x = block(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)

		return x

	def get_sample(self, batch_size):
		return torch.randn(batch_size, 3, 224, 224)

	def get_target(self, batch_size):
		return torch.randint(0, self.fc.out_features, (batch_size,))

	def loss_fn(self, pred, target, *args, **kwargs):
		return torch.nn.functional.cross_entropy(pred, target, *args, **kwargs)


class ResBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(ResBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.relu(out)

		out = self.conv2(out)

		out = out + identity
		out = self.relu(out)

		return out


class SimpleCNN(nn.Module):
	def __init__(self, channels=64):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(
			in_channels=channels, out_channels=channels * 2, kernel_size=3, stride=1, padding=1
		)
		self.conv3 = nn.Conv2d(
			in_channels=channels * 2, out_channels=channels * 4, kernel_size=3, stride=1, padding=1
		)
		self.maxpool = nn.MaxPool2d(2)
		self.relu = nn.ReLU()
		self.avgpool = nn.AvgPool2d(7)
		self.fc1 = nn.Linear(channels * 4 * 8 * 8, 256)
		self.fc2 = nn.Linear(256, 10)

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.maxpool(x)
		x = self.relu(self.conv2(x))
		x = self.maxpool(x)
		x = self.relu(self.conv3(x))
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	def get_sample(self, batch_size):
		return torch.randn(batch_size, 3, 224, 224)

	def get_target(self, batch_size):
		return torch.randint(0, 10, (batch_size,))

	def loss_fn(self, pred, target, *args, **kwargs):
		return torch.nn.functional.cross_entropy(pred, target, *args, **kwargs)


class SimpleAttention(nn.Module):
	def __init__(self, hidden_dim):
		super(SimpleAttention, self).__init__()
		self.hidden_dim = hidden_dim

		self.query = nn.Linear(hidden_dim, hidden_dim)
		self.key = nn.Linear(hidden_dim, hidden_dim)
		self.value = nn.Linear(hidden_dim, hidden_dim)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, inputs):
		# Linear projections
		Q = self.query(inputs)
		K = self.key(inputs)
		V = self.value(inputs)

		# Compute attention scores
		scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
			torch.tensor(self.hidden_dim, dtype=torch.float32)
		)

		# Apply softmax to get attention weights
		attention_weights = self.softmax(scores)

		# Compute the weighted sum of values
		context = torch.matmul(attention_weights, V)

		return context

	def get_sample(self, batch_size):
		return torch.randn(batch_size, 64, self.hidden_dim)

	def get_target(self, batch_size):
		return self.get_sample(batch_size)  # same

	def loss_fn(self, pred, target, *args, **kwargs):
		return torch.nn.functional.mse_loss(pred, target, *args, **kwargs)


class SimpleTransformer(nn.Module):
	def __init__(self, input_dim, hidden_dim, n_blocks=4):
		super(SimpleTransformer, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim

		self.embed = nn.Embedding(input_dim, hidden_dim)
		self.head = nn.Linear(hidden_dim, input_dim)
		self.blocks = []
		for i in range(n_blocks):
			self.blocks.append(
				nn.Sequential(
					SimpleAttention(hidden_dim), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
				)
			)
			self.add_module(f"block_{i}", self.blocks[-1])

	def forward(self, x):
		x = self.embed(x)

		for b in self.blocks:
			x = b(x)

		x = self.head(x)
		return x

	def get_sample(self, batch_size):
		return torch.randint(0, self.input_dim, (batch_size, 64))

	def get_target(self, batch_size):
		return self.get_sample(batch_size)

	def loss_fn(self, pred, target, *args, **kwargs):
		pred = pred.view(-1, self.input_dim)  # flatten seq dim
		target = target.view(-1)
		return torch.nn.functional.cross_entropy(pred, target, *args, **kwargs)
