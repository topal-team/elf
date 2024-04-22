import torch
import os
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
            schedule.append((id_, Operations.FORWARD, {'remat': True}))
            schedule.append((id_, Operations.SEND_FORWARD))
    
    # All backward
    for _ in range(n_micro_batches):
        for id_ in reversed(range(len(placement))):
            schedule.append((id_, Operations.RECV_BACKWARD))
            schedule.append((id_, Operations.BACKWARD, {'remat': True}))
            schedule.append((id_, Operations.SEND_BACKWARD))
    
    assert len(schedule) == n_micro_batches * n_stages * 2 * 3

    return schedule

def generate_1f1b_schedule(placement, n_micro_batches):
    '''
    One Forward One Backward as in PipeDream https://arxiv.org/abs/1806.03377
    '''

    schedule = []
    rank = int(os.getenv("RANK"))

    n_stages = len(placement)
    n_devices = int(max(placement)) + 1
    stages_per_device = n_stages // n_devices

    i = 0
    b_f = 0
    # Warmup phase : each device can compute until the micro batch forward is finished (n_stages), but it can only start after it was forwarded through all the previous layers (rank)
    while i < (stages_per_device * n_micro_batches) and i < (n_stages - rank):
        i += 1
        schedule.append((b_f * n_devices + rank, Operations.RECV_FORWARD))
        schedule.append((b_f * n_devices + rank, Operations.FORWARD))
        schedule.append((b_f * n_devices + rank, Operations.SEND_FORWARD))

        # each layer has time to compute n_devices micro batches before work arrives for the next layer
        # we always prioritize forward on the last possible layer
        # (also, we stop if all micro batches have been computed)
        if (i % n_devices) == 0 or (i % n_micro_batches) == 0:
            b_f = (b_f + 1) % stages_per_device

    # Number of forward passes computed before steady state
    state = i
    b_b = stages_per_device - 1 # last layer first
    
    # Steady state
    while i < (stages_per_device * n_micro_batches):
        i += 1
        schedule.append((b_b * n_devices + rank, Operations.RECV_BACKWARD))
        schedule.append((b_b * n_devices + rank, Operations.BACKWARD))
        schedule.append((b_b * n_devices + rank, Operations.SEND_BACKWARD))

        # Same as before, except that we can compute 2x less micro batches because half of the time is spent doing forwards
        if (i - state) % (n_devices // 2) == 0 or (i - state) % n_micro_batches == 0:
            b_b = (b_b - 1) % stages_per_device

        schedule.append((b_f * n_devices + rank, Operations.RECV_FORWARD))
        schedule.append((b_f * n_devices + rank, Operations.FORWARD))
        schedule.append((b_f * n_devices + rank, Operations.SEND_FORWARD))

        if (i >= n_stages and i % (n_devices // 2) == 0) or (i % n_micro_batches) == 0:
            b_f = (b_f + 1) % stages_per_device

    while i < (stages_per_device * n_micro_batches * 2 - (stages_per_device * n_micro_batches - state)):
        i += 1
        schedule.append((b_b * n_devices + rank, Operations.RECV_BACKWARD))
        schedule.append((b_b * n_devices + rank, Operations.BACKWARD))
        schedule.append((b_b * n_devices + rank, Operations.SEND_BACKWARD))

        # Finish all backwards
        if (i - n_micro_batches - state) % (n_devices // 2) == 0 or (i - n_micro_batches - state) % n_micro_batches == 0:
            b_b = (b_b - 1) % stages_per_device

    # We need to swap send/recv on one out of two devices to avoid deadlocks
    if rank % 2 == 1:
        for i in range(len(schedule) - 1):
            if schedule[i][1] == Operations.SEND_FORWARD and schedule[i + 1][1] == Operations.RECV_BACKWARD or \
               schedule[i][1] == Operations.SEND_BACKWARD and schedule[i + 1][1] == Operations.RECV_FORWARD:
                tmp = schedule[i+1]
                schedule[i+1] = schedule[i]
                schedule[i] = tmp
         

    return schedule

def generate_custom_1f1b_schedule(placement, n_micro_batches):
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

def check_schedule(schedule):
    counts = [0] * len(Operations)
    for (_, op, *_) in schedule:
        counts[int(op)] += 1 # dirty indexing trick
        if int(op) != 0 and op != Operations.RECV_BACKWARD \
                and counts[int(op) - 1] < counts[int(op)]: # Unfortunately, because of the asynchronous comms we cannot really check for recv_backward/send_forward order
            return False, f"The order of operations is wrong !"
    if not all (c == counts[0] for c in counts): 
        return False, "Number of operations does not match !"
    return True, "Schedule is correct :D"

if __name__ == "__main__":
    import torch
    from engine import Operations
    placement = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    for n in range(5):
        schedule = generate_1f1b_schedule(placement, 2**n)
        print(f'[Rank {os.getenv("RANK")}] - {2**n} micro batches : {check_schedule(schedule)} \n')
    # print(f'Rank {os.getenv("RANK")} - {schedule}')
    
else:
    from .engine import Operations
