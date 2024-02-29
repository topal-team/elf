import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import os

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
print(f'GPU {global_rank} set on local device {rank} with world size {world_size}')

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.inputs = None
        self.activations = None

    def forward(self, x):
        # save for backward
        self.inputs = x
        self.activations = self.layer(x)
        return self.activations


def pipelined_forward(layer, sample):
    splits = iter(sample.split(1, dim=0))
    result = []

    for i, x in enumerate(splits):
        # At each step, we manage each device
        for d in range(0, world_size):
            # This process is running the compute
            if d == global_rank:
                # x is already on device 0
                if d != 0 : 
                    dist.recv(x, d - 1)
                    print(f'[GPU {global_rank}] - Received data from {d - 1}, starting compute')
                y = layer(x)
                if d == world_size - 1: # last layer stores result
                    print(f'[GPU {global_rank}] - Finished computing for micro batch {i}')
                    result.append(y)
                else:
                    # send to next layer
                    print(f'[GPU {global_rank}] - Sending my data to {d + 1}')
                    dist.send(y, d + 1)

    print(f'[GPU {global_rank}] - Waiting for all processes..')
    dist.barrier()
    print(f'[GPU {global_rank}] - Done !')

    if result:
        return torch.cat(result, dim=0)
    else:
        return []


def async_pipelined_forward(layer, sample):
    splits = iter(sample.split(1, dim=0))
    result = []
    d = global_rank

    for i, x in enumerate(splits):
        if d != 0:
            status = dist.irecv(x, d - 1)
            status.wait()
            print(f'[GPU {global_rank}] - Received data from rank {d - 1}')
        y = layer(x)
        if d == world_size - 1:
            print(f'[GPU {global_rank}] - Finished compute for micro batch {i}')
            result.append(y)
            continue

        print(f'[GPU {global_rank}] - Sending data to rank {d + 1}')
        dist.isend(y, d + 1)

    print(f'[GPU {global_rank}] - Waiting for all processes..')
    dist.barrier()
    print(f'[GPU {global_rank}] - Done !')
    if result: # only last device has result
        return torch.cat(result, dim=0)
    else:
        return []
        
def pipelined_backward(layers, loss):
    loss.backward() # compute for last block
    layer = layers[global_rank] # should already be on right device / TODO : no it's not

    for d in reversed(range(world_size)):
        # Compute gradients for current block
        if d == global_rank:
            if d != len(layers) - 1:
                dist.recv(loss, d + 1)
                layer.activations.grad.data = loss
            layer.activations.sum().backward()
            if d == 0:
                return layer.inputs.grad.data # finished
        # Send gradients to previous block
        elif d == global_rank - 1:
            dist.send(layer.inputs.grad.data, d)
        

if __name__ == "__main__":
    torch.cuda.set_device(rank) # set before initiating pg or cuda detects multiple processes using same device
    dist.init_process_group(backend="nccl", world_size=world_size)

    layer = LinearBlock(3, 3).to(rank)

    # Send full model to last gpu for comparison later
    states = [None] * world_size if global_rank == world_size - 1 else None
    dist.gather_object(states, layer.state_dict(), dst=world_size - 1)

    sample = torch.randn((4, 3)).to(0) # first layer is on rank 0

    result = pipelined_forward(layer, sample)

    # Only the last rank has the result ; the others can stop
    if global_rank == world_size - 1:
        assert result.shape == sample.shape, f'Error : expected result to have shape {sample.shape}, got {result.shape}'
        
        '''
        target = torch.randn_like(result)
        loss = F.mse_loss(result, target)
        grads = pipelined_backward(layers, loss)
        '''

        # Normal computation for comparison
        # Reconstruct full model on last GPU
        model = nn.Sequential([LayerBlock(3, 3).to(rank) for _ in range(world_size)])
        for i, state in enumerate(states):
            model[i].load_state_dict(state)

        result_full = model(sample)
        result = result
        assert torch.allclose(result, result_full), f'Results are different for cpu and pipelined mode : {result} vs {result_full}'

        '''
        loss = F.mse_loss(result_full, target)
        loss.backward()
        grads_cpu = sample.grad.data
        assert torch.allclose(grads, grads_cpu), f'Gradients are different for cpu and pipelined model : {grads} vs {grads_cpu}'
        '''
    
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
