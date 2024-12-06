"""
This file was borrowed from https://github.com/karpathy/minGPT with modifications.
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import os
import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F


rank = int(os.getenv("RANK", "0"))

logger = logging.getLogger(__name__)


class MyGPTConfig:
	def __init__(self, vocab_size, block_size, **kwargs):
		self.vocab_size = vocab_size
		self.block_size = block_size
		self.embd_pdrop = 0.1
		self.resid_pdrop = 0.1
		self.attn_pdrop = 0.1
		for k, v in kwargs.items():
			setattr(self, k, v)


class GPTConfig:
	"""base GPT config, params common to all GPT versions"""

	embd_pdrop = 0.1
	resid_pdrop = 0.1
	attn_pdrop = 0.1

	def __init__(self, vocab_size, block_size, **kwargs):
		self.vocab_size = vocab_size
		self.block_size = block_size
		for k, v in kwargs.items():
			setattr(self, k, v)


class GPTTinyConfig(GPTConfig):
	n_layer = 12
	n_head = 2
	n_embd = 128


class GPTSmallConfig(GPTConfig):
	"""GPT3-small like network roughly 125M params"""

	n_layer = 12
	n_head = 12
	n_embd = 768


class GPTMediumConfig(GPTConfig):
	"""GPT3-large like network roughly 350M params"""

	n_layer = 24
	n_head = 16
	n_embd = 1024


class GPTLargeConfig(GPTConfig):
	"""GPT3-large like network roughly 760M params"""

	n_layer = 24
	n_head = 16
	n_embd = 1536


class GPTXLConfig(GPTConfig):
	"""GPT3-XL like network roughly 1.3B params"""

	n_layer = 24
	n_head = 24
	n_embd = 2064


class GPTXXLConfig(GPTConfig):
	"""GPT3-XL like network roughly 2.7B params"""

	n_layer = 32
	n_head = 32
	n_embd = 2560


class GPTHanayoConfig(GPTConfig):
	"""~1.6B params"""

	n_layer = 128
	n_head = 16
	n_embd = 1024


class GPTXXXLConfig(GPTConfig):
	"""GPT3-XL like network roughly 6.7B params"""

	n_layer = 32
	n_head = 32
	n_embd = 4096


class GPT13BConfig(GPTConfig):
	"""GPT3-XL like network roughly 13B params"""

	n_layer = 48
	n_head = 48
	n_embd = 5184


class GPT175BConfig(GPTConfig):
	"""GPT3-XL like network roughly 175B params"""

	n_layer = 96
	n_head = 96
	n_embd = 12288


class GPT1TConfig(GPTConfig):
	"""GPT3-XL like network roughly 1T params"""

	n_layer = 128
	n_head = 128
	n_embd = 25600


class Conv1D(nn.Module):
	def __init__(self, nx, nf):
		super().__init__()
		self.nf = nf
		w = torch.empty(nx, nf)
		nn.init.normal_(w, std=0.02)
		self.weight = nn.Parameter(w)
		self.bias = nn.Parameter(torch.zeros(nf))

	def forward(self, x):
		size_out = x.size()[:-1] + (self.nf,)
		x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
		x = x.view(size_out)
		return x


class FeedForward(nn.Module):
	def __init__(self, dropout, d_model=768, nx=768 * 4):
		super().__init__()
		self.c_fc = Conv1D(d_model, nx)
		self.c_proj = Conv1D(nx, d_model)
		self.act = F.gelu
		self.dropout = nn.Dropout(dropout)
		# self.dropout = nn.Identity()

	def forward(self, x):
		return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Attention(nn.Module):
	def __init__(
		self, config, device="cpu", dtype=torch.float32, n_ctx=1024, scale=False, dropout=0.1
	):
		super().__init__()
		self.n_head = config.n_head
		self.d_model = config.n_embd
		self.c_attn = Conv1D(config.n_embd, config.n_embd * 3)
		self.scale = scale
		self.softmax = nn.Softmax(dim=-1)
		self.dropout = nn.Dropout(dropout)
		# self.dropout = nn.Identity()
		self.c_proj = Conv1D(config.n_embd, config.n_embd)

		assert config.n_embd % config.n_head == 0, f"n_embd={config.n_embd}, n_head={config.n_head}"
		d = device if torch.device(device).type == "cuda" else "cpu"
		self.register_buffer(
			"mask",
			torch.tril(torch.ones(config.block_size, config.block_size, device=d, dtype=dtype)).view(
				1, 1, config.block_size, config.block_size
			),
		)
		self.n_head = config.n_head

	def split_heads(self, x):
		"return shape [`batch`, `head`, `sequence`, `features`]"
		new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
		x = x.view(new_shape)
		return x.permute(0, 2, 1, 3)

	def _attn(self, q, k, v, attn_mask=None):
		scores = torch.matmul(q, k.transpose(-2, -1))
		if self.scale:
			scores = scores / math.sqrt(v.size(-1))
		if attn_mask is not None:
			scores = scores + attn_mask
		scores = self.softmax(scores)
		scores = self.dropout(scores)
		outputs = torch.matmul(scores, v)
		return outputs

	def merge_heads(self, x):
		x = x.permute(0, 2, 1, 3).contiguous()
		new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
		return x.view(new_shape)

	def reset_parameters(self):
		for _, m in self.named_modules():
			if isinstance(m, nn.Linear):
				m.reset_parameters()

	def forward(self, x):
		x = self.c_attn(x)  # new `x` shape - `[1,3,2304]`
		q, k, v = x.split(self.d_model, dim=2)
		q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
		scores = torch.matmul(q, k.transpose(-2, -1))
		scores = self.softmax(scores)
		scores = self.dropout(scores)
		out = torch.matmul(scores, v)

		out = self.merge_heads(out)
		out = self.c_proj(out)
		return out


class EmbeddingStem(nn.Module):
	def __init__(self, config, device="cpu", dtype=torch.float32):
		super(EmbeddingStem, self).__init__()

		# input embedding stem

		self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, device=device, dtype=dtype)
		# self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd, device=device, dtype=dtype))
		self.wpe = nn.Embedding(config.block_size, config.n_embd, device=device, dtype=dtype)
		self.drop = nn.Dropout(config.embd_pdrop)
		self.block_size = config.block_size
		self.vocab_size = config.vocab_size
		self.device = device

	def reset_parameters(self):
		self.tok_emb.reset_parameters()

	def forward(self, idx):
		token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
		pos_ids = torch.arange(0, self.block_size, dtype=torch.long, device=self.device).unsqueeze(
			0
		)  # each position maps to a (learnable) vector
		return self.drop(token_embeddings + self.wpe(pos_ids))


class Block(nn.Module):
	"""an unassuming Transformer block"""

	def __init__(self, config, device=None, dtype=torch.float32, wrapper=lambda m: m):
		super(Block, self).__init__()
		self.ln1 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
		self.ln2 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
		self.attn = wrapper(Attention(config, device=device, dtype=dtype))
		self.mlp = nn.Sequential(
			wrapper(nn.Linear(config.n_embd, 4 * config.n_embd, device=device, dtype=dtype)),
			nn.GELU(),
			wrapper(nn.Linear(4 * config.n_embd, config.n_embd, device=device, dtype=dtype)),
			nn.Dropout(config.resid_pdrop),
		)

	def reset_parameters(self):
		self.attn.reset_parameters()
		for _, m in self.named_modules():
			if isinstance(m, nn.LayerNorm) or isinstance(m, nn.Linear):
				m.reset_parameters()

	def forward(self, x):
		x = x + self.attn(self.ln1(x))
		x = x + self.mlp(self.ln2(x))
		return x


class GPT(nn.Module):
	"""the full GPT language model, with a context size of block_size"""

	def __init__(self, config, device="cpu", dtype=torch.float32):
		super(GPT, self).__init__()

		# input embedding stem
		self.emb_stem = EmbeddingStem(config, device=device, dtype=dtype)
		# transformer
		self.blocks = nn.Sequential(
			*[Block(config, device=device, dtype=dtype) for _ in range(config.n_layer)]
		)
		# decoder head
		self.ln_f = nn.LayerNorm(config.n_embd, device=device, dtype=dtype)
		self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False, device=device, dtype=dtype)

	def forward(self, idx):
		x = self.emb_stem(idx)
		x = self.blocks(x)
		x = self.ln_f(x)
		x = self.head(x)
		return x

	def loss_fn(self, logits, targets):
		logits = logits.view(-1, logits.size(-1))
		targets = targets.view(-1)
		return nn.functional.cross_entropy(logits, targets)
