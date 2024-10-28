import torch 
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import datasets
from transformers import AutoTokenizer
from models.llama3 import Llama, ModelArgs
import argparse

# from models.llama3 import Llama, ModelArgs
from models.GPT import GPT, GPT13BConfig, GPTLargeConfig
from pipeline import Pipeline

def init_process(local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank, init_method='env://')
    size = dist.get_world_size()
    if torch.cuda.is_available() and backend=='nccl':
        torch.cuda.set_device(local_rank)
    fn(local_rank, size)

def main()


if __name__ == "__main__":
	local_rank = int(os.getenv("LOCAL_RANK"))
    init_process(local_rank, fn=main, backend='nccl')

	if dist.is_initialized():
		dist.destroy_process_group()
