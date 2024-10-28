'''
nsys profile -t cuda,nvtx,osrt,cublas,cudnn --output=$SCRATCH/nsys-rep/test_p2p/report --force-overwrite true --cuda-memory-usage=true --cudabacktrace=all   --gpu-metrics-device=all --nic-metrics=true --capture-range=cudaProfilerApi --capture-range-end=stop torchrun --nproc-per-node 4  test_p2p.py --executor=torchrun
'''
import torch
import torch.distributed as dist
import os
import random
from functools import partial
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from argparse import ArgumentParser
import nvtx
import torch.cuda.nvtx as tnvtx

torch.backends.cudnn.enabled=True

parser = ArgumentParser()
parser.add_argument("--executor", type=str, default='torchrun')
config = parser.parse_args()

p2p_communication_domain = "P2P_Async_Communications"
collective_communication_domain = "Collective_Communications"
computation_domain = "Computation"


def init_process_for_spawn(rank, size, fn, backend='nccl', port=None):
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
    tensor = torch.ones(1)
    req = None
    if rank == 0:
        tensor *= 2
        _ = tensor @ tensor
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


def run_isendrecv_chain(rank, size):
    torch.cuda.profiler.cudart().cudaProfilerStart()
    tensor = torch.ones((50,10**3), device='cuda')

    if rank != 0:
        # with nvtx.annotate('async receiving', color='green', domain=p2p_communication_domain):
            req = dist.irecv(tensor=tensor, src=rank-1)
            print(f'Rank {rank} received tensor {tensor[:,0]} from Rank {rank-1}')

    if rank != size-1:
        # with nvtx.annotate('compute', color='blue', domain=computation_domain):
            tnvtx.range_push("Tensor manipulations")
            for _ in range(500):
                tensor *= (2 + rank)
                tensor /= torch.norm(tensor)
                tensor = torch.nn.functional.linear(tensor, tensor.T@tensor)
            tnvtx.range_pop()


        # with nvtx.annotate('async sending', color='red', domain=p2p_communication_domain):
            print(f'Rank {rank} sending tensor {tensor[:,0]} to Rank {rank+1}')
            req = dist.isend(tensor=tensor, dst=rank+1)

    req.wait()
    # with nvtx.annotate('all reduce', color='orange', domain=collective_communication_domain):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f'Rank {rank} final tensor value: {tensor[:,0]}')
    torch.cuda.profiler.cudart().cudaProfilerStop()


if __name__=="__main__":
    if config.executor != 'torchrun':
        torch.multiprocessing.set_start_method('spawn', force=True)

    if torch.cuda.is_available():
        print(f"CUDA is available. PyTorch is using CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")


    if config.executor == 'torchrun':
        local_rank = int(os.environ['LOCAL_RANK'])
        init_process(local_rank, fn = run_isendrecv_chain, backend = 'nccl')
        dist.destroy_process_group()

    else:
        size = 4
        processes = []
        port = random.randint(25000, 30000)
        backend = 'nccl'
        mp.spawn(
            init_process_for_spawn,
            args=(size, run_isendrecv_chain, backend, port),
            nprocs=size,  # Number of processes (one per GPU)
            join=True  # Wait for all processes to finish
        )
