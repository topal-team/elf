import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import time
from torch.optim import SGD
import torch.distributed as dist
import numpy as np
# from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.models import resnet50

def time_model(model, sample, n = 20):
    t = 0
    for _ in range(n):
        start = time.time()
        x = model(sample)
        t += time.time() - start

    return t / n, x.shape

class PPresnet50(nn.Module):
    def __init__(self, gpus, split_size = 1):
        super(PPresnet50, self).__init__()
        self.gpus = gpus
        self.split_size = split_size
        model = resnet50()
        self.part1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
        ).to(gpus[0])

        self.part2 = nn.Sequential(
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Flatten(),
            model.fc
        ).to(gpus[1])

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.part1(s_next).to(self.gpus[1])
        out = []
        
        for s_next in splits:
            s_prev = self.part2(s_prev)
            out.append(s_prev)

            s_prev = self.part1(s_next).to(self.gpus[1])

        s_prev = self.part2(s_prev)
        out.append(s_prev)
                
        return torch.cat(out)

class AdaptivePPResnet50(nn.Module):
    def __init__(self, n_waves=1, split_size=1):
        '''
        n_waves: number of interleaves
        '''
        super(AdaptivePPResnet50, self).__init__()
        self.base = resnet50()
        self.n_devices = int(os.environ['WORLD_SIZE'])
        print(f'There are {self.n_devices} devices in use.')
        self.n_waves = n_waves
        self.rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        print(f'My local rank is {self.rank}, global is {self.global_rank}')
        self.splitted = self.split_model()
        self.split_size = split_size

    def split_model(self):
        '''
        Splits the model into sequences on N devices. Each device is affected n_waves parts of the model.
        '''
        times = []
        sample = torch.randn((64, 3, 224, 224), device="cuda:0")
        self.base = self.base.to("cuda:0")
        for name, c in self.base.named_children():
            if name == "fc": sample = sample.view(sample.size(0), -1)
            t, shape = time_model(c, sample)
            times.append(t)
            sample = torch.randn(shape, device="cuda:0")

        exec_time = np.cumsum(times)
        threshold = exec_time[-1] / (self.n_devices * self.n_waves)
        idx = 0
        splitted = [nn.Sequential() for _ in range(self.n_devices * self.n_waves)]
        
        for i, (name, module) in enumerate(self.base.named_children()):
            if i == 0:
                splitted[0].append(module)
                continue

            if name == "fc":
                splitted[idx].append(nn.Flatten())
            splitted[idx].append(module)

            if (exec_time[i] // threshold) > (exec_time[i - 1] // threshold):
                idx = (idx + 1) % self.n_devices # loop over devices

        for w in range(self.n_waves):
            splitted[w * self.n_devices + self.global_rank] = splitted[w * self.n_devices + self.global_rank].to(self.rank)

        return splitted
            

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        ret = []
        inputs = [[] for _ in range(self.n_devices * self.n_waves)]
        # Add all micro batches to queue
        for s in splits:
            inputs[0].append(s.to(0))

        n_microbatches = len(inputs[0])
        
        print(f'{n_microbatches} to compute. (I am rank {self.global_rank})')
        for _ in range(n_microbatches):
            # print(f'New iter for rank {self.global_rank} - {[len(i) for i in inputs]} - Waiting for everyone')
            for w in range(self.n_waves):
                # On every device, we compute our next activation
                idx = w * self.n_devices + self.global_rank # current block
                print(f'[GPU {self.global_rank}] - idx = {idx}, len = {len(inputs) - 1} - {[len(i) for i in inputs]}')
                if inputs[idx]: # not empty
                    # Remove input from queue, compute output and add it to next block's queue
                    output = self.splitted[idx](inputs[idx].pop())
                    
                    if idx == (len(inputs) - 1): # last layer
                        ret.append(output)
                        print(f'New computation finished ; len = {len(ret)}')
                    else:
                        inputs[idx + 1].append(output.to((idx + 1) % self.n_devices))
            dist.barrier() # step
            
        return torch.cat(ret, dim=0)
        

def train(model):
    # setup(rank, world_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-3)

    for _ in range(15):
        optimizer.zero_grad()
        outputs = model(torch.randn((64, 3, 224, 224), device=0))
        labels = torch.randn((64, 1000,), device=outputs.get_device())
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    # cleanup()

if __name__ == "__main__":
    import time
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    world_size = torch.cuda.device_count()
    durations = []
    sizes = np.logspace(1, 6, 6, endpoint=True, base=2)
    # mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    '''
    for size in sizes:
        print(f'Normal PP - Split size = {size}')
        model = PPresnet50(range(2), split_size=int(size))
        start = time.time()
        train(model)
        durations.append(time.time() - start)
    '''
    dist.init_process_group(backend="nccl", world_size=int(os.environ["WORLD_SIZE"]))
    d_adapt = []
    for size in sizes:
        print(f'Adaptive PP - Split size = {size}')
        model = AdaptivePPResnet50(split_size=int(size))
        start = time.time()
        train(model)
        d_adapt.append(time.time() - start)
    dist.destroy_process_group()
    start = time.time()
    train(resnet50().to(0))
    duration_base = time.time() - start

    plt.plot(sizes, durations, label="Pipelined model")
    plt.plot(sizes, d_adapt, label="Adaptive pipelined model")
    plt.plot(sizes, [duration_base] * len(durations), label="Base model")
    plt.xlabel('Split size')
    plt.ylabel('Training time')
    plt.legend()
    plt.savefig("result.png")
    
