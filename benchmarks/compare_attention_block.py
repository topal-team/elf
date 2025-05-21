	
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
# Fused kernels (Flash, Mem-Efficient, CuDNN) require float16 or bfloat16
import sys
sys.path.append(".")
from elf.utils import TimerGPU
from models.simple import TransformerBlock, ChainTransformer
from collections import defaultdict


if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    num_heads = 24
    seq_len_base = 512
    head_dim = 96
    n_blocks = 4
    embed_dim = 2304
    hidden_dim = embed_dim

    stats = defaultdict(list)

    for (sdp_backend, dtype) in [
        (None, torch.float32), (None, torch.float16),
        ("FLASH_ATTENTION", torch.float16), ("FLASH_ATTENTION", torch.bfloat16), 
        ("MATH", torch.float16), ("MATH", torch.bfloat16),]:
        model = TransformerBlock(
                    hidden_dim, 
                    num_heads=num_heads, 
                    dropout=0.1, 
                    ffn_dim=None, 
                    sdp_backend=sdp_backend).to(device).to(dtype)


        print(f"\n=========\n")
        tmp_stats = []
        for seq_len in [seq_len_base*i for i in range(1, 15)]:
            # try:
                # model = (
                #     ChainTransformer(
                #         hidden_dim=hidden_dim,
                #         n_blocks=n_blocks,
                #         seq_len=seq_len,
                #         num_heads=num_heads,
                #         sdp_backend=sdp_backend,
                #     )
                #     .to(device)
                #     .to(dtype)
                # )
                sample = torch.randn(batch_size, seq_len, hidden_dim).to(device).to(dtype)  # keep as long for embedding
                target = torch.randn(batch_size, seq_len, hidden_dim).to(device).to(dtype)
                loss_fn = torch.nn.MSELoss()

                for _ in range(3):
                    output = model(sample)
                    loss = loss_fn(output, target)
                    loss.backward()


                torch.cuda.reset_peak_memory_stats()
                mem_before_forward = torch.cuda.memory_allocated()
                output = model(sample)
                loss = loss_fn(output, target)
                mem_before_backward = torch.cuda.memory_allocated()
                loss.backward()
                mem_peak = torch.cuda.max_memory_allocated()
                print(f"SeqLen: {seq_len}, Dtype: {dtype}, SDPBackend: {sdp_backend}, Before forward: {mem_before_forward/(1024*1024)}, Before backward: {mem_before_backward/(1024*1024)}, Peak: {mem_peak/(1024*1024)}, (BeforeBw-BeforeFw)/BeforeFw: {(mem_before_backward-mem_before_forward)/mem_before_forward}, (BeforeBw-BeforeFw)/seqlen: {(mem_before_backward-mem_before_forward)/(1024*1024*seq_len)}")

                tmp_stats.append([seq_len, mem_before_forward/(1024*1024), mem_before_backward/(1024*1024), mem_peak/(1024*1024)])
            # except:
            #     print(f'SeqLen: {seq_len}, Dtype: {dtype}, SDPBackend: {sdp_backend}: OOM most probably!')
            #     tmp_stats.append([seq_len, None, None, None])

        # stats[(sdp_backend, dtype)] = tmp_stats
    
    print(stats)