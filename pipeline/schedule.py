'''
Manage different types of schedule. All scheduling algorithm (GPipe, 1f1b, ...) are defined here.
A schedule is a list of operations (see Operation in ``graph.py``) that will be executed in order by each device.
Every rank should generate the entire schedule for all ranks, in order to detect and fix cycles/deadlocks.
'''

import torch

import logging
logger = logging.getLogger("schedule")

def generate_afab_schedule(placement, n_micro_batches, prefetching = False, **options):
    '''
    All Forward All Backward as in GPipe https://arxiv.org/abs/1811.06965
    '''
    schedule = []
    n_stages = len(placement)
    n_devices = max(placement) + 1

    # All forward
    for rank in range(n_devices):
        ids = [i for i in range(len(placement)) if placement[i] == rank]
        for i in range(n_micro_batches):
            for id_ in ids:
                schedule.append(Operation(id_, i, OperationType.RECV_FORWARD, rank, options))
                schedule.append(Operation(id_, i, OperationType.FORWARD, rank, options))
                schedule.append(Operation(id_, i, OperationType.SEND_FORWARD, rank, options))
        
        # All backward
        for i in range(n_micro_batches):
            for id_ in reversed(ids):
                schedule.append(Operation(id_, i, OperationType.RECV_BACKWARD, rank, options))
                schedule.append(Operation(id_, i, OperationType.BACKWARD, rank, options))
                schedule.append(Operation(id_, i, OperationType.SEND_BACKWARD, rank, options))
    
    assert len(schedule) == n_micro_batches * n_stages * 2 * 3
    if prefetching: return enable_prefetching(schedule)
    return schedule

def generate_1f1b_schedule(placement, n_micro_batches, prefetching = False, **options):
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
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.RECV_FORWARD, rank, options))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.FORWARD, rank, options))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.SEND_FORWARD, rank, options))
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
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.RECV_BACKWARD, rank, options))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.BACKWARD, rank, options))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.SEND_BACKWARD, rank, options))
            bwds[b_b] += 1

            # Same as before, except that we can compute 2x less micro batches because half of the time is spent doing forwards
            if (i - state) % (n_devices // 2) == 0 or (i - state) % n_micro_batches == 0:
                b_b = (b_b - 1) % stages_per_device

            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.RECV_FORWARD, rank, options))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.FORWARD, rank, options))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], OperationType.SEND_FORWARD, rank, options))
            fwds[b_f] += 1

            if (i >= n_stages and i % (n_devices // 2) == 0) or (i % n_micro_batches) == 0:
                b_f = (b_f + 1) % stages_per_device

        while i < (stages_per_device * n_micro_batches * 2 - (stages_per_device * n_micro_batches - state)):
            i += 1
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.RECV_BACKWARD, rank, options))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.BACKWARD, rank, options))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], OperationType.SEND_BACKWARD, rank, options))
            bwds[b_b] += 1

            # Finish all backwards
            if (i - n_micro_batches - state) % (n_devices // 2) == 0 or (i - n_micro_batches - state) % n_micro_batches == 0:
                b_b = (b_b - 1) % stages_per_device

    if prefetching: return enable_prefetching(schedule)
    return schedule

def generate_hanayo_schedule(placement, n_micro_batches, prefetching = False, **options):
    schedule = []
    n_devices = max(placement) + 1
    n_stages = len(placement)
    n_waves = n_stages // n_devices

    for rank in range(n_devices):
        ids = [i for i in range(len(placement)) if placement[i] == rank]
        # Warmup Phase
        i = 0
        while i < n_micro_batches and i < (n_devices - rank):
            schedule.append(Operation(ids[0], i, OperationType.RECV_FORWARD, rank, options))
            schedule.append(Operation(ids[0], i, OperationType.FORWARD, rank, options))
            schedule.append(Operation(ids[0], i, OperationType.SEND_FORWARD, rank, options))
            i += 1
        
        # Steady
        for i in range(n_micro_batches * n_waves):
            if i % 2 == 0 and (i//2 < n_micro_batches):
                schedule.append(Operation(ids[1], i // 2, OperationType.RECV_FORWARD, rank, options))
                schedule.append(Operation(ids[1], i // 2, OperationType.FORWARD, rank, options))
                schedule.append(Operation(ids[1], i // 2, OperationType.SEND_FORWARD, rank, options))
            elif i//2 + n_devices - rank < n_micro_batches:
                schedule.append(Operation(ids[0], (i//2) + n_devices - rank, OperationType.RECV_FORWARD, rank, options))
                schedule.append(Operation(ids[0], (i//2) + n_devices - rank, OperationType.FORWARD, rank, options))
                schedule.append(Operation(ids[0], (i//2) + n_devices - rank, OperationType.SEND_FORWARD, rank, options))
            elif (i//2) < n_micro_batches:
                schedule.append(Operation(ids[1], (i - 2 * rank) // 2, OperationType.RECV_BACKWARD, rank, options))
                schedule.append(Operation(ids[1], (i - 2 * rank) // 2, OperationType.BACKWARD, rank, options))
                schedule.append(Operation(ids[1], (i - 2 * rank) // 2, OperationType.SEND_BACKWARD, rank, options))

        # Cooldown
        todo = n_micro_batches * (n_waves - 1) - (n_devices - rank)
        for i in range(n_micro_batches * n_waves):
            if i % 2 == 0:
                schedule.append(Operation(ids[0], i // 2, OperationType.RECV_BACKWARD, rank, options))
                schedule.append(Operation(ids[0], i // 2, OperationType.BACKWARD, rank, options))
                schedule.append(Operation(ids[0], i // 2, OperationType.SEND_BACKWARD, rank, options))
            elif todo > 0:
                schedule.append(Operation(ids[1], n_micro_batches - todo, OperationType.RECV_BACKWARD, rank, options))
                schedule.append(Operation(ids[1], n_micro_batches - todo, OperationType.BACKWARD, rank, options))
                schedule.append(Operation(ids[1], n_micro_batches - todo, OperationType.SEND_BACKWARD, rank, options))
                todo -= 1

    if prefetching: return reorder_operations(schedule)
    return schedule

if __name__ == "__main__":
    from graph import *
    placement = [0, 1, 2, 3, 3, 2, 1, 0]
    schedule = generate_hanayo_schedule(placement, 4)
    # graph = graph_from_schedule(schedule)
    # if cycles := find_cycles(graph):
    #     logger.warning(f'Found potential deadlocks in the schedule !')
    #     for c in cycles: print(c)
    #     print()
    
    # new_schedule = schedule_from_graph(graph)
    for rank in range(max(placement) + 1):
        local_schedule = list(filter(lambda op: op.rank == rank, schedule))
        # local_new_schedule = list(filter(lambda op: op.block_id == rank, new_schedule))
        print(f'Rank {rank}')
        print(local_schedule, "\n")
        # print(local_new_schedule)
        # assert local_schedule == local_new_schedule
else:
    from .graph import *
