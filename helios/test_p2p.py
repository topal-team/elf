import torch
import torch.distributed as dist
import os
import random
from functools import partial
import torch.multiprocessing as mp
from torch.multiprocessing import Process


def init_process2(rank, size, fn, backend='nccl', port=None):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = f'{port}'
    dist.init_process_group(backend, rank=rank, world_size=size)

    if torch.cuda.is_available() and backend == 'nccl':
        torch.cuda.set_device(rank)  # Set the GPU corresponding to this rank
        print(f"Process {rank} using GPU {torch.cuda.current_device()}")

    fn(rank, size)
    dist.destroy_process_group()

def init_process(local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank, init_method='env://')
    size = dist.get_world_size()
    if torch.cuda.is_available() and backend=='nccl':
        torch.cuda.set_device(local_rank)
    fn(local_rank, size)
    


def run_isendrecv(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor.to(rank), dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        tensor = tensor.to(rank)
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    print('Rank ', rank, ' has data ', tensor[0])
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])


def run_isendrecv_4(rank, size):
    tensor = torch.zeros(1).to(rank)

    if rank == 1:
        # Rank 1 initializes the tensor and sends to Rank 2
        tensor += 1
        print(f'Rank {rank} sending tensor {tensor.item()} to Rank 2')
        req = dist.isend(tensor=tensor, dst=2)
        req.wait()

    elif rank == 2:
        # Rank 2 receives from Rank 1, then sends to Rank 3
        req = dist.irecv(tensor=tensor, src=1)
        req.wait()
        print(f'Rank {rank} received tensor {tensor.item()} from Rank 1')
        tensor += 1
        req = dist.isend(tensor=tensor, dst=3)
        req.wait()

    elif rank == 3:
        # Rank 3 receives from Rank 2, then sends to Rank 4
        req = dist.irecv(tensor=tensor, src=2)
        req.wait()
        print(f'Rank {rank} received tensor {tensor.item()} from Rank 2')
        tensor += 1
        req = dist.isend(tensor=tensor, dst=4)
        req.wait()

    elif rank == 4:
        # Rank 4 receives from Rank 3
        req = dist.irecv(tensor=tensor, src=3)
        req.wait()
        print(f'Rank {rank} received tensor {tensor.item()} from Rank 3')

    print(f'Rank {rank} final tensor value: {tensor.item()}')

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    if torch.cuda.is_available():
        print(f"CUDA is available. PyTorch is using CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")

    # local_rank = int(os.environ['LOCAL_RANK'])
    # init_process(local_rank, fn = run_isendrecv_4, backend = 'nccl')
    # dist.destroy_process_group()

    size = 4
    processes = []
    port = random.randint(25000, 30000)
    backend = 'nccl'
    # for rank in range(size):
    #     p = Process(target=init_process2, args=(rank, size, run_isendrecv, backend, port))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()


        # Start multiple processes using torch.multiprocessing.spawn
    mp.spawn(
        init_process2,
        args=(size, run_isendrecv_4, backend, port),
        nprocs=size,  # Number of processes (one per GPU)
        join=True  # Wait for all processes to finish
    )
