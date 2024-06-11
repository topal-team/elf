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
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
# import rockmate
# import rkgb

from torch.distributed.fsdp.wrap import wrap

rank = int(os.getenv("RANK", "0"))

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
except ImportError:
    from fairscale.nn.checkpoint import checkpoint_wrapper

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPTSmallConfig(GPTConfig):
    """ GPT3-small like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class GPTMediumConfig(GPTConfig):
    """ GPT3-large like network roughly 350M params """
    n_layer = 24
    n_head = 16
    n_embd = 1024

class GPTLargeConfig(GPTConfig):
    """ GPT3-large like network roughly 760M params """
    n_layer = 24
    n_head = 16
    n_embd = 1536

class GPTXLConfig(GPTConfig):
    """ GPT3-XL like network roughly 1.3B params """
    n_layer = 24
    n_head = 24
    n_embd = 2064

class GPTXXLConfig(GPTConfig):
    """ GPT3-XL like network roughly 2.7B params """
    n_layer = 32
    n_head = 32
    n_embd = 2560

class GPTHanayoConfig(GPTConfig):
    n_layer = 128
    n_head = 16
    n_embd = 1024
    
class GPTXXXLConfig(GPTConfig):
    """ GPT3-XL like network roughly 6.7B params """
    n_layer = 32
    n_head = 32
    n_embd = 4096


class GPT13BConfig(GPTConfig):
    """ GPT3-XL like network roughly 13B params """
    n_layer = 48
    n_head = 48
    n_embd = 5184


class GPT175BConfig(GPTConfig):
    """ GPT3-XL like network roughly 175B params """
    n_layer = 96
    n_head = 96
    n_embd = 12288

class GPT1TConfig(GPTConfig):
    """ GPT3-XL like network roughly 1T params """
    n_layer = 128
    n_head = 128
    n_embd = 25600

def module_wrapper(module, fsdp=False, activation="noop"):
    if not fsdp:
        return module

    if activation == "noop":
        return wrap(module)
    elif activation == "checkpoint":
        return wrap(checkpoint_wrapper(module))
    elif activation == "offload":
        return wrap(checkpoint_wrapper(module, offload_to_cpu=True))
    else:
        raise ValueError(f"Unrecognized activation mode {activation}")


# class CausalSelfAttention(nn.Module):
#     """
#     A vanilla multi-head masked self-attention layer with a projection at the end.
#     It is possible to use torch.nn.MultiheadAttention here but I am including an
#     explicit implementation here to show that there is nothing too scary here.
#     """

#     def __init__(self, config, device="cpu", dtype=torch.float32):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0, f"n_embd={config.n_embd}, n_head={config.n_head}"
#         # key, query, value projections for all heads
#         self.key = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
#         self.query = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
#         self.value = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
#         # regularization
#         self.attn_drop = nn.Dropout(config.attn_pdrop)
#         self.resid_drop = nn.Dropout(config.resid_pdrop)
#         # output projection
#         self.proj = nn.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
#         # causal mask to ensure that attention is only applied to the left in the input sequence
#         # TODO: leave buffer on CPU for now, until we can do meta_tensor.to_empty()
#         d = device if torch.device(device).type == "cuda" else "cpu"
#         self.register_buffer(
#             "mask",
#             torch.tril(torch.ones(config.block_size, config.block_size, device=d, dtype=dtype))
#                  .view(1, 1, config.block_size, config.block_size)
#         )
#         self.n_head = config.n_head

#     def reset_parameters(self):
#         for _, m in self.named_modules():
#             if isinstance(m, nn.Linear):
#                 m.reset_parameters()

#     def forward(self, x, layer_past=None):
#         B, T, C = x.size()

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

#         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
#         # att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = torch.matmul(q, k.transpose(-2, -1))
#         att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#         y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

#         # output projection
#         y = self.resid_drop(self.proj(y))
#         return y

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
        self,
        config,
        device="cpu", 
        dtype=torch.float32,
        n_ctx=1024,
        scale=False,
        dropout=0.1,
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
            torch.tril(torch.ones(config.block_size, config.block_size, device=d, dtype=dtype))
                 .view(1, 1, config.block_size, config.block_size)
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
        nd, ns = scores.size(-2), scores.size(-1)
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
        # out      = self._attn(q, k, v)
        scores = torch.matmul(q, k.transpose(-2, -1))
        # if self.scale: scores = scores/math.sqrt(v.size(-1))
        nd, ns = scores.size(-2), scores.size(-1)
        # if attn_mask is not None: scores = scores + attn_mask
        scores = self.softmax(scores)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)

        out = self.merge_heads(out)
        out = self.c_proj(out)
        return out

# class GPT2_input(nn.Module):
#     def __init__(self, n_ctx=1024, d_model=768, vcb_sz=50257, dropout=0.1):
#         super(GPT2_input, self).__init__()
#         block = TransformerBlock(d_model=d_model, n_head=12, dropout=dropout)
#         # self.h = _get_clones(block, nlayers)
#         self.wte = nn.Embedding(vcb_sz, d_model)
#         self.wpe = nn.Embedding(n_ctx, d_model)
#         self.drop = nn.Dropout(dropout)
#         # self.drop = nn.Identity()
#         self.ln_f = LayerNorm(d_model)
#         self.out = nn.Linear(d_model, vcb_sz, bias=False)
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.init_weights()

#     def init_weights(self):
#         self.out.weight = self.wte.weight
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if (
#                 isinstance(module, (nn.Linear, Conv1D))
#                 and module.bias is not None
#             ):
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def forward(
#         self, src, labels=None, pos_ids=None, return_inp=False, dropout=0.1
#     ):
#         if pos_ids is None:
#             pos_ids = torch.arange(
#                 0, src.size(-1), dtype=torch.long, device=self.wpe.weight.device
#             ).unsqueeze(0)
#         inp = self.drop((self.wte(src) + self.wpe(pos_ids)))
#         return inp

class EmbeddingStem(nn.Module):

    def __init__(self, config, device="cpu", dtype=torch.float32):

        super(EmbeddingStem, self).__init__()

        # input embedding stem

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, device=device, dtype=dtype)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd, device=device, dtype=dtype))
        self.wpe = nn.Embedding(config.block_size, config.n_embd, device=device, dtype=dtype)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size

    def reset_parameters(self):
        self.tok_emb.reset_parameters()

    def forward(self, idx):

        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        pos_ids = torch.arange(
                0, idx.size(-1), dtype=torch.long, device=self.wpe.weight.device
            ).unsqueeze(0) # each position maps to a (learnable) vector
        return self.drop(token_embeddings + self.wpe(pos_ids))


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(
        self,
        config,
        device=None,
        dtype=torch.float32,
        wrapper=lambda m : m,
        version="pytorch",
        cpu_offload=False,
    ):
        super(Block, self).__init__()
        if version == "pytorch" or not cpu_offload:
            self.ln1 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
            self.ln2 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
            self.attn = wrapper(Attention(config, device=device, dtype=dtype))
            self.mlp = nn.Sequential(
                wrapper(nn.Linear(config.n_embd, 4 * config.n_embd, device=device, dtype=dtype)),
                nn.GELU(),
                wrapper(nn.Linear(4 * config.n_embd, config.n_embd, device=device, dtype=dtype)),
                nn.Dropout(config.resid_pdrop),
            )
        else:
            print("fairscale fsdp for block")
            self.ln1 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype).cpu())
            self.ln2 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype).cpu())
            self.attn = wrapper(Attention(config, device=device, dtype=dtype).cpu())
            self.mlp = nn.Sequential(
                wrapper(nn.Linear(config.n_embd, 4 * config.n_embd, device=device, dtype=dtype).cpu()),
                nn.GELU(),
                wrapper(nn.Linear(4 * config.n_embd, config.n_embd, device=device, dtype=dtype).cpu()),
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
    """  the full GPT language model, with a context size of block_size """

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

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, idx):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.head(x)
        return x


def configure_optimizers(model, train_config):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('pos_emb') and isinstance(m, EmbeddingStem):
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters() if "_fsdp_wrapped_module" not in pn}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
    return optimizer


def sequential_gpt(config, devices, dtype=torch.float32):
    """
    Returns an ``nn.Sequential`` of GPT model balanced across the given devices.
    N.B.: this function does not dedup devices.
    """
    # put all layers into a list
    emb_stem = EmbeddingStem(config, device="meta", dtype=dtype)
    blocks = [Block(config, device="meta", dtype=dtype) for _ in range(config.n_layer)]
    ln_f = nn.LayerNorm(config.n_embd, device="meta", dtype=dtype)
    head = nn.Linear(config.n_embd, config.vocab_size, bias=False, device="meta", dtype=dtype)

    layers = [emb_stem, *blocks, ln_f, head]

    # partition layers into the given devices
    def numel(layer):
        return sum([p.numel() for p in layer.parameters()])

    total_numel = sum([numel(layer) for layer in layers])
    phase_numel = total_numel // len(devices)
    delim_numel = phase_numel
    accum_numel = 0

    # seal one pipeline phase when its numel is larger than phase_numel
    phases = [[]]
    for layer in layers:
        phases[-1].append(layer)
        accum_numel += numel(layer)
        if accum_numel > delim_numel:
            delim_numel += phase_numel
            phases.append([])

    # pack all remaining layers into the last phase
    while len(phases) > len(devices):
        phases[-2].extend(phases[-1])
        phases.pop()

    for i, phase in enumerate(phases):
        for layer in phase:
            layer.to_empty(device=torch.device(devices[i])).reset_parameters()

    # create nn.Sequential
    return nn.Sequential(*[nn.Sequential(*phase) for phase in phases])

def sequential_gpt_2(model, devices, dtype=torch.float32):
    """
    Returns an ``nn.Sequential`` of GPT model balanced across the given devices.
    N.B.: this function does not dedup devices.
    """
    layers = []
    for module in model.children():
        if isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential):
            layers.extend(module)
        else:
            layers.append(module)

    # partition layers into the given devices
    def numel(layer):
        return sum([p.numel() for p in layer.parameters()])

    total_numel = sum([numel(layer) for layer in layers])
    phase_numel = total_numel // len(devices)
    delim_numel = phase_numel
    accum_numel = 0

    # seal one pipeline phase when its numel is larger than phase_numel
    phases = [[]]
    for layer in layers:
        phases[-1].append(layer)
        accum_numel += numel(layer)
        if accum_numel > delim_numel:
            delim_numel += phase_numel
            phases.append([])

    # pack all remaining layers into the last phase
    while len(phases) > len(devices):
        phases[-2].extend(phases[-1])
        phases.pop()

   
    for i, phase in enumerate(phases):
        for layer in phase:
            layer.to_empty(device=torch.device(devices[i]))
    # create nn.Sequential
    return nn.Sequential(*[nn.Sequential(*phase) for phase in phases])
    

def PipelineGPT(model, devices, vocab_size, input_shape, checkpoint = False, budget = None):

    # operator = FNO(n_modes=(16, 16), hidden_channels=64, in_channels=3, out_channels=1)
    
    input = None
    rkMods = []

    sequential_model = sequential_gpt_2(model, devices)

    if checkpoint and budget != None :
        for module in sequential_model.children():
            device  = next(module.named_parameters())[1].device

            if input is None:
                input = torch.randint(0, vocab_size, input_shape)

            print(input.shape)

            # list_solver = [rockmate.solvers.HILP()]
            list_solver = [rockmate.solvers.TwRemat()]
            max_size_S_graph_for_no_partitioning = 0
            partitioners = [rkgb.Ptools.Partitioner_seq(sub_partitioner=rkgb.Ptools.Partitioner())]

            
            input = input.to(device)
            rkMod = rockmate.HRockmate(
                    module, input, budget, 
                    list_solvers=list_solver, 
                    partitioners=partitioners,
                    # solve_sched = False,
                    max_size_S_graph_for_no_partitioning=max_size_S_graph_for_no_partitioning
                )

            with torch.no_grad():
                output = module(input)
                input_shape = output.shape
                input = output.detach()

            torch.cuda.empty_cache()

            rkMod.solve_sched(budget, rec=False)
            rkMod.get_compiled_fct()

            rkMods.append(rkMod)

        del input
        
        return nn.Sequential(*rkMods)
    else:
        
        return sequential_model



class ShardedGPT(nn.Module):
    def __init__(self, config, device="cpu", dtype=torch.float32, activation="noop", version="pytorch", cpu_offload=False):
        super().__init__()

        if version == "pytorch" or not cpu_offload:
            wrapper = partial(module_wrapper, fsdp=True, activation=activation)

            # input embedding stem
            self.emb_stem = wrap(EmbeddingStem(config, device=device, dtype=dtype))
            # transformer
            self.blocks = nn.Sequential(
                *[wrapper(Block(config, device=device, dtype=dtype, wrapper=wrap)) for _ in range(config.n_layer)]
            )
            # decoder head
            self.ln_f = wrap(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
            self.head = wrap(nn.Linear(config.n_embd, config.vocab_size, bias=False, device=device, dtype=dtype))

            if rank == 0:
                print("number of parameters:", sum(p.numel() for p in self.parameters()))
        else:
            print("fariscale fsdp for shardedGPT")
            wrapper = partial(module_wrapper, fsdp=True, activation=activation)
            # input embedding stem
            self.emb_stem = wrap(EmbeddingStem(config, device=device, dtype=dtype).cpu())
            # transformer
            self.blocks = nn.Sequential(
                *[wrapper(Block(config, device=device, dtype=dtype, wrapper=wrap, version=version, cpu_offload=True).cpu()) for _ in range(config.n_layer)]
            )
            # decoder head
            self.ln_f = wrap(nn.LayerNorm(config.n_embd, device=device, dtype=dtype).cpu())
            self.head = wrap(nn.Linear(config.n_embd, config.vocab_size, bias=False, device=device, dtype=dtype).cpu())

    def forward(self, idx):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)
