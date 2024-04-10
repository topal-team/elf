from .engine import Operations

import math
import logging
logger = logging.getLogger("schedule")

def generate_afab_schedule(placement, n_micro_batches):
    '''
    All Forward All Backward as in GPipe https://arxiv.org/abs/1811.06965
    Supports any model placement
    '''
    schedule = []
    n_stages = len(placement)

    # All forward
    for _ in range(n_micro_batches):
        for id_ in range(n_stages):
            schedule.append((id_, Operations.RECV_FORWARD))
            schedule.append((id_, Operations.FORWARD))
            schedule.append((id_, Operations.SEND_FORWARD))
    
    # All backward
    for _ in range(n_micro_batches):
        for id_ in reversed(range(len(placement))):
            schedule.append((id_, Operations.RECV_BACKWARD))
            schedule.append((id_, Operations.BACKWARD))
            schedule.append((id_, Operations.SEND_BACKWARD))
    
    assert len(schedule) == n_micro_batches * n_stages * 2 * 3

    return schedule

def generate_1f1b_schedule(placement, n_micro_batches):
    '''
    One Forward One Backward as in PipeDream https://arxiv.org/abs/1806.03377
    '''
    schedule = []
    n_stages = len(placement)
    n_devices = int(max(placement)) + 1
    stages_per_device = len(placement) // n_devices
    
    for d in range(n_devices):
        # we assume blocks are placed sequentially among every device, e.g. [0, 1, 2, 0, 1, 2]
        # and there are at least as many micro batches as there are devices
        offset = d * (stages_per_device * n_micro_batches * 2 * 3)

        for i in range(n_micro_batches * stages_per_device):
            b = ((i // min(n_micro_batches, n_devices)) % stages_per_device)
            schedule.append((b * n_devices + d, Operations.RECV_FORWARD))
            schedule.append((b * n_devices + d, Operations.FORWARD))
            schedule.append((b * n_devices + d, Operations.SEND_FORWARD))

        magic = n_stages + (n_devices - d - 1) - d
        magic = min(magic, n_micro_batches * stages_per_device)
        for i in range(n_micro_batches * stages_per_device):
            b = stages_per_device - ((i // min(n_micro_batches, math.ceil(n_devices / 2))) % stages_per_device) - 1
            schedule.insert(offset + (magic * 3), (b * n_devices + d, Operations.RECV_BACKWARD))
            schedule.insert(offset + (magic * 3) + 1, (b * n_devices + d, Operations.BACKWARD))
            schedule.insert(offset + (magic * 3) + 2, (b * n_devices + d, Operations.SEND_BACKWARD))
            magic += 4

    assert len(schedule) == n_micro_batches * n_stages * 2 * 3

    return schedule

if __name__ == "__main__":
    import torch
    placement = torch.tensor([0, 1, 0, 1])
    schedule = generate_1f1b_schedule(placement, 1)
    for rank in range(placement.max().item() + 1):
        actions = [(id_, op) for id_, op in schedule if placement[id_] == rank]
        print(f'Rank {rank} - {actions}\n')

    print(schedule)
