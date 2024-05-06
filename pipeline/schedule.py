import torch
import os
import math
import logging
logger = logging.getLogger("schedule")

def reorder_operations(operations):
    # Define the target and source operations
    target_ops = {Operations.FORWARD, Operations.BACKWARD}
    source_ops = {Operations.RECV_FORWARD, Operations.RECV_BACKWARD}
    
    # We need to iterate while keeping track of indexes because we'll modify the list
    i = 0
    while i < len(operations):
        _, op, *_ = operations[i]
        if op in target_ops:
            # Look for the next source operation
            for j in range(i + 1, len(operations)):
                _, op_next, *_ = operations[j]
                if op_next in source_ops:
                    # Move found operation to just before the current one
                    operations.insert(i, operations.pop(j))
                    i += 1
                    break
        i += 1
    return operations

def generate_afab_schedule(placement, n_micro_batches, *options, prefetching = False):
    '''
    All Forward All Backward as in GPipe https://arxiv.org/abs/1811.06965
    Supports any model placement
    '''
    schedule = []

    rank = int(os.getenv("RANK"))
    ids = [i for i in range(len(placement)) if placement[i] == rank]

    # All forward
    for i in range(n_micro_batches):
        for id_ in ids:
            schedule.append((id_, Operations.RECV_FORWARD, i, *options))
            schedule.append((id_, Operations.FORWARD, i, *options))
            schedule.append((id_, Operations.SEND_FORWARD, i, *options))
    
    # All backward
    for i in range(n_micro_batches):
        for id_ in reversed(ids):
            schedule.append((id_, Operations.RECV_BACKWARD, i, *options))
            schedule.append((id_, Operations.BACKWARD, i, *options))
            schedule.append((id_, Operations.SEND_BACKWARD, i, *options))
    
    assert len(schedule) == n_micro_batches * len(ids) * 2 * 3
    if prefetching: return reorder_operations(schedule)
    return schedule

def generate_1f1b_schedule(placement, n_micro_batches, prefetching = False):
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
         
    if prefetching: return reorder_operations(schedule)
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
    placement = torch.tensor([0, 1, 2, 3])
    schedule = generate_afab_schedule(placement, 4)
    schedule = reorder_operations(schedule)
    print(f'[Rank {os.getenv("RANK")}] - {4} micro batches : {check_schedule(schedule)}\n')
    print(f'Rank {os.getenv("RANK")} - {schedule}')
    
else:
    from .engine import Operations
