import torch

import logging
logger = logging.getLogger("schedule")

def generate_afab_schedule(placement, n_micro_batches, *options, prefetching = False):
    '''
    All Forward All Backward as in GPipe https://arxiv.org/abs/1811.06965
    Supports any model placement
    '''
    schedule = []
    n_stages = len(placement)
    n_devices = max(placement) + 1

    # All forward
    for rank in range(n_devices):
        ids = [i for i in range(len(placement)) if placement[i] == rank]
        for i in range(n_micro_batches):
            for id_ in ids:
                schedule.append(Operation(id_, i, OperationType.RECV_FORWARD, rank, *options))
                schedule.append(Operation(id_, i, OperationType.FORWARD, rank, *options))
                schedule.append(Operation(id_, i, OperationType.SEND_FORWARD, rank, *options))
        
        # All backward
        for i in range(n_micro_batches):
            for id_ in reversed(ids):
                schedule.append(Operation(id_, i, OperationType.RECV_BACKWARD, rank, *options))
                schedule.append(Operation(id_, i, OperationType.BACKWARD, rank, *options))
                schedule.append(Operation(id_, i, OperationType.SEND_BACKWARD, rank, *options))
    
    assert len(schedule) == n_micro_batches * n_stages * 2 * 3
    if prefetching: return reorder_operations(schedule)
    return schedule

def generate_1f1b_schedule(placement, n_micro_batches, prefetching = False):
    '''
    One Forward One Backward as in PipeDream https://arxiv.org/abs/1806.03377
    '''
    schedule = []
    n_stages = len(placement)
    n_devices = int(max(placement)) + 1
    stages_per_device = n_stages // n_devices

    for rank in range(n_devices):
        fwds = [0] * stages_per_device
        bwds = [0] * stages_per_device

        i = 0
        b_f = 0
        # Warmup phase : each device can compute until the micro batch forward is finished (n_stages), but it can only start after it was forwarded through all the previous layers (rank)
        while i < (stages_per_device * n_micro_batches) and i < (n_stages - rank):
            i += 1
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.RECV_FORWARD, rank))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.FORWARD, rank))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.SEND_FORWARD, rank))
            fwds[b_f] += 1

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
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.RECV_BACKWARD, rank))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.BACKWARD, rank))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.SEND_BACKWARD, rank))
            bwds[b_b] += 1

            # Same as before, except that we can compute 2x less micro batches because half of the time is spent doing forwards
            if (i - state) % (n_devices // 2) == 0 or (i - state) % n_micro_batches == 0:
                b_b = (b_b - 1) % stages_per_device

            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.RECV_FORWARD, rank))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.FORWARD, rank))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.SEND_FORWARD, rank))
            fwds[b_f] += 1

            if (i >= n_stages and i % (n_devices // 2) == 0) or (i % n_micro_batches) == 0:
                b_f = (b_f + 1) % stages_per_device

        while i < (stages_per_device * n_micro_batches * 2 - (stages_per_device * n_micro_batches - state)):
            i += 1
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.RECV_BACKWARD, rank))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.BACKWARD, rank))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.SEND_BACKWARD, rank))
            bwds[b_b] += 1

            # Finish all backwards
            if (i - n_micro_batches - state) % (n_devices // 2) == 0 or (i - n_micro_batches - state) % n_micro_batches == 0:
                b_b = (b_b - 1) % stages_per_device

    if prefetching: return reorder_operations(schedule)
    return schedule

def check_schedule(schedule):
    counts = [0] * len(OperationType)
    for (_, op, *_) in schedule:
        counts[int(op)] += 1 # dirty indexing trick
        if int(op) != 0 and op != OperationType.RECV_BACKWARD \
                and counts[int(op) - 1] < counts[int(op)]: # Unfortunately, because of the asynchronous comms we cannot really check for recv_backward/send_forward order
            return False, f"The order of operations is wrong !"
    if not all (c == counts[0] for c in counts): 
        return False, "Number of operations does not match !"
    return True, "Schedule is correct :D"

if __name__ == "__main__":
    import torch
    from graph import *
    placement = torch.tensor([0, 1, 2, 3])
    schedule = generate_1f1b_schedule(placement, 4, prefetching = False)
    graph = graph_from_schedule(schedule)
    if cycles := find_cycles(graph):
        logger.warning(f'Found potential deadlocks in the schedule !')
        for c in cycles: print(c)
        print()
    
    new_schedule = schedule_from_graph(graph)
    for rank in range(len(placement)):
        local_schedule = list(filter(lambda op: op.block_id == rank, schedule))
        local_new_schedule = list(filter(lambda op: op.block_id == rank, new_schedule))
        print(f'\nBlock {rank}\n')
        print(local_schedule)
        print()
        print(local_new_schedule)
        assert local_schedule == local_new_schedule
else:
    from .graph import *
