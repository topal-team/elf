import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet50

class FakeDataset(Dataset):
    def __getitem__(self, idx):
        return (torch.randn((3, 224, 224)), \
                torch.randint(0, 1000, (1,)).item())

    def __len__(self):
        return 1000

def setup():
    print(f'Setting process {os.environ["RANK"]} on device {os.environ["LOCAL_RANK"]}')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group('nccl')

    
def cleanup():
    dist.destroy_process_group()

def run_epoch(model, data, optimizer, epoch):
    data.sampler.set_epoch(epoch)
    for img, label in data:
        optimizer.zero_grad()
        img = img.cuda()
        label = label.cuda()
        # print(f'[GPU{os.environ["RANK"]}] Image and label are on devices : {img.get_device(), label.get_device()}')
        outputs = model(img)
        loss = torch.nn.functional.cross_entropy(outputs, label)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    setup()
    model = resnet50().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])
    print(f'[GPU{os.environ["RANK"]}] Model placed on device {torch.cuda.current_device()}')

    dataset = FakeDataset()
    loader = DataLoader(dataset, batch_size=128, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))
    # print(f'GPU{os.environ["RANK"]} - Dataset and Loader created')

    for e in range(10):
        print(f'GPU{os.environ["RANK"]} : epoch {e}')
        run_epoch(ddp_model, loader, optimizer, e)

    cleanup()
