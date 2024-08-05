import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from argparse import ArgumentParser


class Dummy(Dataset):
    def __init__(self, n = 16) -> None:
        super().__init__()
        self.n = n

    def __getitem__(self, index):
        assert index < self.n
        return torch.full((4,), index, dtype = torch.float32, device = torch.cuda.current_device())
    
    def __len__(self):
        return self.n

def single(model):
    print(f'Beginning on single gpu')
    print(list(model.parameters()))
    model = model.cuda()
    data = Dummy()
    optimizer = Adam(model.parameters())
    loader = DataLoader(data, batch_size = 4)

    for d in loader:
        optimizer.zero_grad()
        loss = model(d).sum()
        loss.backward()
        optimizer.step()

    print(f'\nEnd of epoch on single gpu')
    print(list(model.parameters()))

def multi(model):
    rank = int(os.getenv('RANK'))
    dist.init_process_group(backend = 'nccl')
    torch.cuda.set_device(rank)
    
    print(f'Beginning on multi gpu')
    print(list(model.parameters()))

    model = model.cuda()
    ddp = DDP(model, device_ids = [rank])
    data = Dummy()
    optimizer = Adam(ddp.parameters())
    loader = DataLoader(data, batch_size = 4)
    for d in loader:
        optimizer.zero_grad()
        loss = ddp(d).sum()
        loss.backward()
        optimizer.step()

    
    print(f'\nEnd of epoch on multi gpu')
    print(list(model.parameters()))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', '-m', type=str)
    args = parser.parse_args()

    torch.manual_seed(47)
    model = nn.Linear(4, 1)

    if args.mode == 'single':
        single(model)
    elif args.mode == 'multi':
        multi(model)
    else:
        print("?")
