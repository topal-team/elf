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
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


# -----------------------------------------------------------------------------
# ModelArgs


@dataclass
class ModelArgs:
	dim: int = 4096
	n_layers: int = 32
	n_heads: int = 32
	vocab_size: int = -1
	ffn_dim: int = 14336
	norm_eps: float = 1e-5
	rope_theta: float = 500000
	max_batch_size: int = 32
	max_seq_len: int = 8192
	flash: bool = False  # use flash attention?

	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)


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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	t = torch.arange(end, device=freqs.device, dtype=torch.float32)
	freqs = torch.outer(t, freqs)
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
	return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
	ndim = x.ndim
	assert 0 <= 1 < ndim
	assert freqs_cis.shape == (x.shape[1], x.shape[-1])
	shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
	return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
	xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
	xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
	freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
	xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
	xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
	return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()
		self.flash = args.flash  # use flash attention?
		self.n_heads = args.n_heads
		self.head_dim = args.dim // args.n_heads

		self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
		self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
		self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
		self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

		self.register_buffer("freqs_cis", args.freqs_cis.clone().detach())

	def forward(self, x: torch.Tensor):
		bsz, seqlen, _ = x.shape
		xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
		xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
		xk = xk.reshape(bsz, seqlen, self.n_heads, self.head_dim)
		xv = xv.reshape(bsz, seqlen, self.n_heads, self.head_dim)
		freqs_cis = self.freqs_cis[:seqlen]
		xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
		xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))

		# with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
		if self.flash:
			with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
				output = F.scaled_dot_product_attention(xq, xk, xv)
		else:
			output = F.scaled_dot_product_attention(xq, xk, xv)

		output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
		proj = self.wo(output)
		return proj


class FeedForward(nn.Module):
	def __init__(self, dim: int, hidden_dim: int):
		super().__init__()
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
		self.feed_forward = FeedForward(dim=args.dim, hidden_dim=args.ffn_dim)
		self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
		self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

	def forward(self, x: torch.Tensor):
		h = x + self.attention(self.attention_norm(x))
		out = h + self.feed_forward(self.ffn_norm(h))
		return out


class Llama(nn.Module):
	def __init__(self, params: ModelArgs):
		super().__init__()
		self.params = params
		self.vocab_size = params.vocab_size
		self.n_layers = params.n_layers

		params.freqs_cis = precompute_freqs_cis(
			params.dim // params.n_heads, params.max_seq_len * 2, params.rope_theta
		)

		self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
		self.layers = nn.ModuleList(TransformerBlock(params) for _ in range(params.n_layers))
		self.norm = RMSNorm(params.dim, eps=params.norm_eps)
		self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

	def forward(self, inputs: torch.Tensor):
		h = self.tok_embeddings(inputs)
		for layer in self.layers:
			h = layer(h)
		h = self.norm(h)
		return self.output(h)

	def loss(self, outputs, targets, ignore_index=-100):
		logits = outputs.float()
		loss = F.cross_entropy(
			input=logits.transpose(1, 2), target=targets, reduction="mean", ignore_index=ignore_index
		)
		return loss
