"""
nano-Llama 3.1
Simpler version you can just forward on 1 GPU, without torchrun.
Changes:
- replace ColumnParallelLinear -> Linear
- replace RowParallelLinear -> Linear
- replace VocabParallelEmbedding -> Embedding

Run example:

python llama31.py \
	--ckpt_dir llama-models/models/llama3_1/Meta-Llama-3.1-8B \
	--tokenizer_path llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model
"""

from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# ModelArgs


@dataclass
class ModelArgs:
	dim: int = 4096
	n_layers: int = 32
	n_heads: int = 32
	n_kv_heads: Optional[int] = 4
	vocab_size: int = -1
	multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
	ffn_dim_multiplier: Optional[float] = 2
	norm_eps: float = 1e-5
	rope_theta: float = 500000
	use_scaled_rope: bool = False
	max_batch_size: int = 32
	max_seq_len: int = 2048
	flash: bool = False  # use flash attention?

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)
		self.n_kv_heads = self.n_heads
		self.n_kv_heads = self.n_heads
		self.n_heads = self.n_heads


# -----------------------------------------------------------------------------
# Transformer


class RMSNorm(torch.nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))

	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x):
		output = self._norm(x.float()).type_as(x)
		return output * self.weight


def apply_scaling(freqs: torch.Tensor):
	scale_factor = 8
	new_freqs = freqs / scale_factor
	return new_freqs


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	t = torch.arange(end, device=freqs.device, dtype=torch.float32)
	freqs = apply_scaling(freqs)
	freqs = torch.outer(t, freqs)
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
	freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
	return freqs_cis_real


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
	# shape gymnastics let's go
	# x is (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
	# freqs_cis is (seq_len, head_dim/2, 2), e.g. (8, 64, 2)
	xshaped = x.float().reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
	# xshaped is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
	freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
	# freqs_cis becomes (1, seqlen, 1, head_dim/2, 2), e.g. (1, 8, 1, 64, 2)
	x_out2 = torch.stack(
		[
			xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
			xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
		],
		-1,
	)
	# x_out2 at this point is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
	x_out2 = x_out2.flatten(3)
	# x_out2 is now (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
	return x_out2.type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
	bs, slen, n_kv_heads, head_dim = x.shape
	return (
		x[:, :, :, None, :]
		.expand(bs, slen, n_kv_heads, n_rep, head_dim)
		.reshape(bs, slen, n_kv_heads * n_rep, head_dim)
	)


class Attention(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()
		self.flash = args.flash  # use flash attention?
		self.n_kv_heads = args.n_heads
		model_parallel_size = 1  # AK: model parallel size is 1 for 1 GPU
		self.n_local_heads = args.n_heads // model_parallel_size
		self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
		self.n_rep = self.n_local_heads // self.n_local_kv_heads
		self.head_dim = args.dim // args.n_heads

		self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
		self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
		self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
		self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

	def forward(self, x: torch.Tensor, max_seqlen: Optional[int]):
		bsz, seqlen, _ = x.shape
		xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
		xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
		xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
		xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
		# xq = apply_rotary_emb(xq, freqs_cis)
		# xk = apply_rotary_emb(xk, freqs_cis)
		xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
		xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
		xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))
		mask = torch.full((max_seqlen, max_seqlen), float("-inf"))
		mask = torch.triu(mask, diagonal=1)
		mask = mask.type_as(xq)
		output = F.scaled_dot_product_attention(xq, xk, xv, mask)
		output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
		proj = self.wo(output)
		return proj


class FeedForward(nn.Module):
	def __init__(
		self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float] = 1
	):
		super().__init__()
		hidden_dim = int(2 * hidden_dim / 3)
		hidden_dim = int(ffn_dim_multiplier * hidden_dim)
		hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
		self.w1 = nn.Linear(dim, hidden_dim, bias=False)
		self.w2 = nn.Linear(hidden_dim, dim, bias=False)
		self.w3 = nn.Linear(dim, hidden_dim, bias=False)

	def forward(self, x):
		return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()
		self.n_heads = args.n_heads
		self.dim = args.dim
		self.head_dim = args.dim // args.n_heads
		self.attention = Attention(args)
		self.feed_forward = FeedForward(
			dim=args.dim,
			hidden_dim=4 * args.dim,
			multiple_of=args.multiple_of,
			ffn_dim_multiplier=args.ffn_dim_multiplier,
		)
		self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
		self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

	def forward(self, x: torch.Tensor, max_seqlen: Optional[int]):
		h = x + self.attention(self.attention_norm(x), max_seqlen)
		out = h + self.feed_forward(self.ffn_norm(h))
		return out


class Llama(nn.Module):
	def __init__(self, params: ModelArgs):
		super().__init__()
		self.params = params
		self.vocab_size = params.vocab_size
		self.n_layers = params.n_layers
		self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

		self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
		self.layers = nn.ModuleList(TransformerBlock(params) for _ in range(params.n_layers))
		self.norm = RMSNorm(params.dim, eps=params.norm_eps)
		self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

		self.freqs_cis = precompute_freqs_cis(
			params.dim // params.n_heads,
			params.max_seq_len * 2,
			params.rope_theta,
			params.use_scaled_rope,
		)

	def forward(self, inputs: torch.Tensor):
		h = self.tok_embeddings(inputs)
		# freqs_cis = self.freqs_cis[:self.params.max_seq_len]
		for layer in self.layers:
			h = layer(h, self.params.max_seq_len)
		h = self.norm(h)
		return self.output(h)

	def loss(self, outputs, targets, ignore_index=-100):
		logits = outputs.float()
		loss = F.cross_entropy(
			input=logits.transpose(1, 2), target=targets, reduction="mean", ignore_index=ignore_index
		)
		return loss
