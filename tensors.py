import torch
import torch.distributed as dist
import torch.profiler
import os

def main(world_size, rank, global_rank):
    # Processes send to next rank and receive from previous one
    src = (global_rank - 1) % world_size
    dst = (global_rank + 1) % world_size


    # Synchronous
    x = torch.randint(0, 10, (3,)).to(rank)
    for r in range(world_size):
        if r == global_rank:
            print(f'[GPU {global_rank}] - Sending {x} synchronously')
            dist.send(x, dst)
        elif r == src:
            y = torch.zeros_like(x).to(rank)
            dist.recv(y, src)
            print(f'[GPU {global_rank}] - Received {y} synchronously')

    # Asynchronous
    x = torch.randint(0, 10, (3,)).to(rank)
    print(f'[GPU {global_rank}] - Sending {x} asynchronously')
    dist.isend(x, dst)

    y = torch.zeros_like(x).to(rank)
    status = dist.irecv(y, src)
    status.wait()
    print(f'[GPU {global_rank}] - Received {y} asynchronously')

if __name__ == "__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    torch.cuda.set_device(rank)

    dist.init_process_group(backend="nccl", world_size=world_size)

    with profiler.profile(
            activities = [profiler.ProfilerActivity.CUDA],
            profile_memory=True
    ) as p:
        main(world_size, rank, global_rank)

    print(f'[GPU {global_rank}] :', p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
