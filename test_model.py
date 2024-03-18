import torch
import torch.nn as nn

def load_full_model(size, path="test-model.pt"):
    '''
    Loads the sequential model stored in `path`, and returns the `size` first layers.
    '''
    model = torch.load(path)
    new_model = nn.Sequential(*list(model.children())[:size])
    return nn.Sequential(new_model)

def load_parts_model(placement, global_rank, path="test-model.pt"):
    '''
    Loads the sequential model stored in `path`, and returns the layers corresponding to `global_rank` in the model placement.
    '''
    indices = [idx for idx, p in enumerate(placement) if global_rank == p]
    model = torch.load(path)
    children = list(model.children())
    blocks = [children[idx] for idx in indices]
    return blocks

if __name__ == "__main__":
    all_layers = [nn.Linear(3, 3, bias=False) for i in range(10)]

    model = nn.Sequential(*all_layers)
    torch.save(model, "test-model.pt")