import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
	from flash_attn import flash_attn_func  # pyright: ignore[reportMissingImports]
except ImportError:
	from torch.nn.attention import SDPBackend, sdpa_kernel

	# If flash_attn is not installed, use SDPA from torch instead
	def flash_attn_func(q, k, v, dropout_p):
		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)
		with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
			out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, enable_gqa=True)
		return out.transpose(1, 2)


# Rotary embeddings from https://github.com/meta-llama/llama/blob/main/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 1000000.0):
	"""
	Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

	This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
	and the end index 'end'. The 'theta' parameter scales the frequencies.
	The returned tensor contains complex values in complex64 data type.

	Args:
		dim (int): Dimension of the frequency tensor.
		end (int): End index for precomputing frequencies.
		theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

	Returns:
		torch.Tensor: Precomputed frequency tensor with complex exponentials.
	"""
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	t = torch.arange(end, device=freqs.device)  # type: ignore
	freqs = torch.outer(t, freqs).float()  # type: ignore
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
	return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
	"""
	Reshape frequency tensor for broadcasting it with another tensor.

	This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
	for the purpose of broadcasting the frequency tensor during element-wise operations.

	Args:
		freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
		x (torch.Tensor): Target tensor for broadcasting compatibility.

	Returns:
		torch.Tensor: Reshaped frequency tensor.
	"""
	ndim = x.ndim
	assert 0 <= 1 < ndim
	assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
		f"{freqs_cis.shape} != {x.shape[1], x.shape[-1]}"
	)
	shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
	return freqs_cis.view(*shape)


def apply_rotary_emb(
	xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Apply rotary embeddings to input tensors using the given frequency tensor.

	This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
	frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
	is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
	returned as real tensors.

	Args:
		xq (torch.Tensor): Query tensor to apply rotary embeddings.
		xk (torch.Tensor): Key tensor to apply rotary embeddings.
		freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

	Returns:
		Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
	"""
	xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
	xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
	freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
	xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
	xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
	return xq_out.type_as(xq), xk_out.type_as(xk)


class QwenAttention(nn.Module):
	def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout, freqs_cis):
		super(QwenAttention, self).__init__()

		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.num_kv_heads = num_kv_heads
		self.head_dim = hidden_dim // num_heads
		self.q_dim = self.num_heads * self.head_dim
		self.kv_dim = self.num_kv_heads * self.head_dim
		self.dropout = dropout

		# Query projection
		self.q_proj = nn.Linear(
			hidden_dim, self.q_dim, bias=False
		)  # Qwen2.5 has qkv bias, but not in Qwen3

		# Key and Value projections (potentially fewer heads for GQA)
		self.k_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)
		self.v_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)

		# Output projection
		self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

		self.qnorm = nn.RMSNorm(hidden_dim)
		self.knorm = nn.RMSNorm(self.kv_dim)

		self.freqs_cis = freqs_cis
		# use different buffer per block for auto partitioning, BUT this makes .to(dtype) cast this buffer to real, making the computation wrong
		# self.register_buffer("freqs_cis", freqs_cis)

	def forward(self, x):
		xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
		# QK Norm as in https://arxiv.org/pdf/2302.05442
		xq = self.qnorm(xq)
		xk = self.knorm(xk)
		xq = xq.reshape(
			xq.shape[0], xq.shape[1], self.num_heads, -1
		)  # (batch, seqlen, num_heads, head_dim)
		xk = xk.reshape(
			xk.shape[0], xk.shape[1], self.num_kv_heads, -1
		)  # (batch, seqlen, num_kv_heads, head_dim_kv)
		xv = xv.reshape(
			xv.shape[0], xv.shape[1], self.num_kv_heads, -1
		)  # (batch, seqlen, num_kv_heads, head_dim_kv)
		xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis)
		y = flash_attn_func(xq, xk, xv, dropout_p=self.dropout)
		y = y.reshape(y.shape[0], y.shape[1], self.hidden_dim)  # (batch, seqlen, hidden_dim)
		return self.o_proj(y)


class SwiGLU(nn.Module):
	def __init__(self, hidden_dim, ffn_dim):
		super(SwiGLU, self).__init__()
		self.hidden_dim = hidden_dim
		self.ffn_dim = ffn_dim

		self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
		self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
		self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
		self.swish = nn.SiLU()

	def forward(self, x):
		gate = self.gate_proj(x)
		up = self.up_proj(x)
		swish_up = self.swish(up)
		return self.down_proj(gate * swish_up)


class QwenBlock(nn.Module):
	def __init__(self, hidden_dim, num_heads, num_kv_heads, dropout, ffn_dim, freqs_cis):
		super(QwenBlock, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.num_kv_heads = num_kv_heads or num_heads
		self.dropout = dropout
		self.ffn_dim = ffn_dim

		self.norm1 = nn.RMSNorm(hidden_dim)
		self.norm2 = nn.RMSNorm(hidden_dim)

		self.attn = QwenAttention(hidden_dim, num_heads, num_kv_heads, dropout, freqs_cis)
		self.ffn = SwiGLU(hidden_dim, ffn_dim)

	def forward(self, x):
		# Pre-norm as in https://arxiv.org/pdf/2305.14858
		x = x + self.attn(self.norm1(x))
		x = x + self.ffn(self.norm2(x))
		return x


class Qwen(nn.Module):
	def __init__(
		self, input_dim, hidden_dim, n_blocks, seq_len, num_heads, num_kv_heads, dropout, ffn_dim
	):
		super(Qwen, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.n_blocks = n_blocks
		self.seq_len = seq_len

		# https://github.com/pytorch/pytorch/issues/131328
		# Fixed in 2.9.0
		device = (
			torch.get_default_device()
			if torch.__version__ >= "2.9.0"
			else (torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
		)

		with torch.device(device):
			self.freqs_cis = precompute_freqs_cis(hidden_dim // num_heads, seq_len)

		self.embed = nn.Embedding(input_dim, hidden_dim)
		self.blocks = [
			QwenBlock(hidden_dim, num_heads, num_kv_heads, dropout, ffn_dim, self.freqs_cis)
			for _ in range(n_blocks)
		]
		self.blocks = nn.Sequential(*self.blocks)

		self.head = nn.Linear(hidden_dim, input_dim, bias=False)

	def forward(self, x):
		x = self.embed(x)
		x = self.blocks(x)
		x = self.head(x)
		return x

	def loss_fn(self, outputs, targets, ignore_index=-100):
		logits = outputs.float()
		loss = F.cross_entropy(
			input=logits.transpose(1, 2), target=targets, reduction="mean", ignore_index=ignore_index
		)
		return loss

	# We keep dtype parameter for compatibility with other models, but it is ignored
	def get_sample(self, batch_size, dtype=torch.int64, device=None):
		return torch.randint(0, self.input_dim, (batch_size, self.seq_len), device=device)

	def get_target(self, batch_size, dtype=torch.int64, device=None):
		return self.get_sample(batch_size, dtype, device)
