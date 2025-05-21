import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
# Fused kernels (Flash, Mem-Efficient, CuDNN) require float16 or bfloat16

"""
This is a model zoo for tests and benchmarks.
Most model implement a get_sample(), get_target(), and loss_fn() method for better interoperability.
"""


class SimpleResNet(nn.Module):
	"""
	Regular, almost homogeneous ResNet with constant channel and image dimensions.
	"""

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
	"""
	Util for SimpleResNet.
	"""

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
	"""
	Simple classification CNN with two convolutions and a ReLU activation.
	"""

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
	"""
	Simple attention layer with a single head.
	"""

	def __init__(self, hidden_dim):
		super(SimpleAttention, self).__init__()
		self.hidden_dim = hidden_dim

		self.query = nn.Linear(hidden_dim, hidden_dim)
		self.key = nn.Linear(hidden_dim, hidden_dim)
		self.value = nn.Linear(hidden_dim, hidden_dim)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, inputs):
		# Linear projections
		Q = self.query(inputs).unsqueeze(1)
		K = self.key(inputs).unsqueeze(1)
		V = self.value(inputs).unsqueeze(1)

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


class SimpleFastAttention(nn.Module):
	"""
	Simple attention layer with a single head.
	sdp_backend: can be 'FLASH_ATTENTION' or 'MATH'
	"""

	def __init__(self, hidden_dim, sdp_backend="FLASH_ATTENTION"):
		super(SimpleFastAttention, self).__init__()
		self.hidden_dim = hidden_dim

		self.query = nn.Linear(hidden_dim, hidden_dim)
		self.key = nn.Linear(hidden_dim, hidden_dim)
		self.value = nn.Linear(hidden_dim, hidden_dim)
		self.softmax = nn.Softmax(dim=-1)
		self.sdp_backend = sdp_backend

	def forward(self, inputs):
		# Linear projections
		Q = self.query(inputs).unsqueeze(1)
		K = self.key(inputs).unsqueeze(1)
		V = self.value(inputs).unsqueeze(1)
		with sdpa_kernel(backends=[SDPBackend.__dict__[sdp_backend]]):
			context = F.scaled_dot_product_attention(Q, K, V).squeeze(1)
			return context

	def get_sample(self, batch_size):
		return torch.randn(batch_size, 64, self.hidden_dim)

	def get_target(self, batch_size):
		return self.get_sample(batch_size)  # same

	def loss_fn(self, pred, target, *args, **kwargs):
		return torch.nn.functional.mse_loss(pred, target, *args, **kwargs)


class SimpleTransformer(nn.Module):
	"""
	Simple transformer with a single head.
	ffn_dim is optional and defaults to hidden_dim * 4.
	sdp_backend can be 'FLASH_ATTENTION', 'MATH' or None
	"""

	def __init__(self, input_dim, hidden_dim, n_blocks=4, seq_len=64, ffn_dim=None, sdp_backend=None):
		super(SimpleTransformer, self).__init__()

		ffn_dim = ffn_dim or hidden_dim * 4

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.seq_len = seq_len

		self.embed = nn.Embedding(input_dim, hidden_dim)
		self.head = nn.Linear(hidden_dim, input_dim)
		self.blocks = []
		for i in range(n_blocks):
			if sdp_backend is not None:
				self.blocks.append(
					nn.Sequential(
						SimpleFastAttention(hidden_dim, sdp_backend),
						nn.LayerNorm(hidden_dim),
						FeedForward(hidden_dim, ffn_dim),
						nn.LayerNorm(hidden_dim),
					)
				)
			else:
				self.blocks.append(
					nn.Sequential(
						SimpleAttention(hidden_dim),
						nn.LayerNorm(hidden_dim),
						FeedForward(hidden_dim, ffn_dim),
						nn.LayerNorm(hidden_dim),
					)
				)
			self.add_module(f"block_{i}", self.blocks[-1])

	def forward(self, x):
		x = self.embed(x)
		x = x.to(self.head.weight.dtype)

		for b in self.blocks:
			x = b(x)

		x = self.head(x)
		return x

	def get_sample(self, batch_size):
		return torch.randint(0, self.input_dim, (batch_size, self.seq_len))

	def get_target(self, batch_size):
		return self.get_sample(batch_size)

	def loss_fn(self, pred, target, *args, **kwargs):
		pred = pred.view(-1, self.input_dim)  # flatten seq dim
		target = target.view(-1)
		return torch.nn.functional.cross_entropy(pred, target, *args, **kwargs)


class FeedForward(nn.Module):
	"""
	Util for transformers.
	"""

	def __init__(self, dim, ffn_dim):
		super(FeedForward, self).__init__()
		self.fc1 = nn.Linear(dim, ffn_dim)
		self.gelu = nn.GELU()
		self.fc2 = nn.Linear(ffn_dim, dim)

	def forward(self, x):
		return self.fc2(self.gelu(self.fc1(x)))


class Attention(nn.Module):
	"""
	Util for transformers.
	"""

	def __init__(self, dim, dropout=0.1):
		super().__init__()
		self.dim = dim
		self.scale = 1.0 / math.sqrt(self.dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, q, k, v):
		# Attention scores
		attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
		attn = F.softmax(attn, dim=-1)
		attn = self.dropout(attn)
		return torch.matmul(attn, v)


class FastAttention(nn.Module):
	def __init__(self, dim, dropout=0.1, sdp_backend="FLASH_ATTENTION"):
		super().__init__()
		self.dim = dim
		self.scale = 1.0 / math.sqrt(self.dim)
		self.dropout = nn.Dropout(dropout)
		self.sdp_backend = sdp_backend

	def forward(self, q, k, v):
		with sdpa_kernel(backends=[SDPBackend.__dict__[self.sdp_backend]]):
			context = F.scaled_dot_product_attention(q, k, v)
			return context


class MultiHeadAttention(nn.Module):
	"""
	Util for transformers.
	"""

	def __init__(self, dim, num_heads=4, dropout=0.1, sdp_backend="FLASH_ATTENTION"):
		super().__init__()
		assert dim % num_heads == 0, "dim must be divisible by num_heads"

		self.dim = dim
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		# Single large linear layers for Q,K,V
		self.q_proj = nn.Linear(dim, dim)
		self.k_proj = nn.Linear(dim, dim)
		self.v_proj = nn.Linear(dim, dim)

		if sdp_backend is not None:
			self.attn = FastAttention(self.head_dim, dropout, sdp_backend=sdp_backend)
		else:
			self.attn = Attention(self.head_dim, dropout)

	def forward(self, x):
		batch_size, seq_len, _ = x.shape

		# Project and split heads: (batch, seq, dim) -> (batch, seq, num_heads, head_dim)
		q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
		k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
		v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

		# Transpose for attention: (batch, num_heads, seq, head_dim)
		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)

		out = self.attn(q, k, v)

		# Reshape back: (batch, seq, dim)
		out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)

		return out

	def get_sample(self, batch_size):
		return torch.randn(batch_size, 64, self.hidden_dim)

	def get_target(self, batch_size):
		return self.get_sample(batch_size)  # same

	def loss_fn(self, pred, target, *args, **kwargs):
		return torch.nn.functional.mse_loss(pred, target, *args, **kwargs)


class TransformerBlock(nn.Module):
	"""
	Util for transformers.
	"""

	def __init__(self, dim, num_heads=4, dropout=0.1, ffn_dim=None, sdp_backend=None):
		super().__init__()
		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		self.attn = MultiHeadAttention(dim, num_heads, dropout, sdp_backend)
		self.proj = nn.Linear(dim, dim)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		ffn_dim = ffn_dim or dim * 4
		self.mlp = FeedForward(dim, ffn_dim)

	def forward(self, x):
		residual = x
		y = self.norm1(x)

		y = self.attn(y)
		y = self.proj(y)
		y = self.dropout1(y)
		y = residual + y
		residual = y

		y = self.norm2(y)
		y = self.mlp(y)
		y = self.dropout2(y)
		y = residual + y

		return y


class FullTransformer(nn.Module):
	"""
	Full decoder-only transformer architecture, with embedding and output head.
	ffn_dim is optional and defaults to hidden_dim * 4.
	"""

	def __init__(
		self,
		input_dim,
		hidden_dim,
		n_blocks=4,
		seq_len=64,
		num_heads=4,
		dropout=0.1,
		ffn_dim=None,
		sdp_backend=None,
	):
		super(FullTransformer, self).__init__()

		ffn_dim = ffn_dim or hidden_dim * 4

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.seq_len = seq_len

		self.embed = nn.Embedding(input_dim, hidden_dim)
		self.head = nn.Linear(hidden_dim, input_dim)
		self.blocks = []
		for i in range(n_blocks):
			self.blocks.append(TransformerBlock(hidden_dim, num_heads, dropout, ffn_dim, sdp_backend))
			self.add_module(f"block_{i}", self.blocks[-1])

	def forward(self, x):
		x = self.embed(x)

		x = x.to(self.head.weight.dtype)

		for b in self.blocks:
			x = b(x)

		x = self.head(x)
		return x

	def get_sample(self, batch_size):
		return torch.randint(0, self.input_dim, (batch_size, self.seq_len))

	def get_target(self, batch_size):
		return self.get_sample(batch_size)

	def loss_fn(self, pred, target, *args, **kwargs):
		pred = pred.view(-1, self.input_dim)  # flatten seq dim
		target = target.view(-1)
		return torch.nn.functional.cross_entropy(pred, target, *args, **kwargs)


class ChainTransformer(nn.Module):
	"""
	Homogeneous chain of transformer blocks.
	ffn_dim is optional and defaults to hidden_dim * 4.
	"""

	def __init__(
		self,
		hidden_dim,
		n_blocks=4,
		seq_len=64,
		num_heads=4,
		dropout=0.1,
		ffn_dim=None,
		sdp_backend=None,
	):
		super(ChainTransformer, self).__init__()

		ffn_dim = ffn_dim or hidden_dim * 4

		self.hidden_dim = hidden_dim
		self.seq_len = seq_len
		self.blocks = []
		for i in range(n_blocks):
			self.blocks.append(TransformerBlock(hidden_dim, num_heads, dropout, ffn_dim, sdp_backend))
			self.add_module(f"block_{i}", self.blocks[-1])

	def forward(self, x):
		for b in self.blocks:
			x = b(x)
		return x

	def get_sample(self, batch_size):
		return torch.randn(batch_size, self.seq_len, self.hidden_dim)

	def get_target(self, batch_size):
		return self.get_sample(batch_size)

	def loss_fn(self, pred, target, *args, **kwargs):
		return torch.nn.functional.mse_loss(pred, target, *args, **kwargs)


if __name__ == "__main__":
	import sys

	sys.path.append("../")
	from elf.utils import TimerGPU

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	sdp_backend = "FLASH_ATTENTION"
	if sdp_backend in ["FLASH_ATTENTION", "MATH"]:
		dtype = torch.float16
	else:
		dtype = torch.float32

	batch_size = 16
	num_heads = 1
	seq_len = 1024
	head_dim = 64
	n_blocks = 4
	embed_dim = num_heads * head_dim
	hidden_dim = embed_dim


	model = (
		FullTransformer(
			input_dim=embed_dim,
			hidden_dim=hidden_dim,
			n_blocks=n_blocks,
			seq_len=seq_len,
			num_heads=num_heads,
			sdp_backend=sdp_backend,
		)
		.to(device)
		.to(dtype)
	)

	g = torch.fx.symbolic_trace(model).graph
	g.print_tabular()

	sample = model.get_sample(batch_size).to(device)  # keep as long for embedding
	target = model.get_target(batch_size).to(device)

	for _ in range(3):
		output = model(sample)
		loss = model.loss_fn(output, target)
		loss.backward()

	with TimerGPU() as timer:
		for _ in range(10):
			output = model(sample)
			loss = model.loss_fn(output, target)
			loss.backward()
	print(f"Forward and backward pass completed through FullTransformer Completed!: {timer.time()}")

	model = (
		ChainTransformer(
			hidden_dim=hidden_dim,
			n_blocks=n_blocks,
			seq_len=seq_len,
			num_heads=num_heads,
			sdp_backend=sdp_backend,
		)
		.to(device)
		.to(dtype)
	)

	g = torch.fx.symbolic_trace(model).graph
	g.print_tabular()

	sample = model.get_sample(batch_size).to(device).to(dtype)  # keep as long for embedding
	target = model.get_target(batch_size).to(device).to(dtype)

	for _ in range(3):
		output = model(sample)
		loss = model.loss_fn(output, target)
		loss.backward()


	with TimerGPU() as timer:
		for _ in range(10):
			output = model(sample)
			loss = model.loss_fn(output, target)
			loss.backward()
	print(f"Forward and backward pass completed: {timer.time()}")
