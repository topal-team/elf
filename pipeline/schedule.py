import torch
import os
import math

from enum import Enum

import logging
logger = logging.getLogger("schedule")

class Operations(Enum):
    RECV_FORWARD = 0
    FORWARD = 1
    SEND_FORWARD = 2
    RECV_BACKWARD = 3
    BACKWARD = 4
    SEND_BACKWARD = 5
    
    def __repr__(self) -> str:
        return self.name.lower()
    
    def __int__(self) -> int:
        return self.value

class Operation():
    def __init__(self, block_id, mb_id, op, options = {}):
        self.block_id = block_id
        self.op = op
        self.mb_id = mb_id
        self.options = options
        self.dependencies = []

    def add_dependency(self, node):
        self.dependencies.append(node)

    def __repr__(self) -> str:
        return f'{self.block_id}:{repr(self.op)}({self.mb_id})'

def graph_from_schedule(schedule):
    # For each one, add forward/backward dependencies (from the sequential nature)
    for operation in schedule:
        match operation.op:
            case Operations.SEND_BACKWARD:
                deps = [op for op in schedule if op.mb_id == operation.mb_id and (\
                        (op.op == Operations.BACKWARD and op.block_id == operation.block_id) or \
                            (op.op == Operations.RECV_BACKWARD and op.block_id == operation.block_id - 1))]
                for d in deps:
                    operation.add_dependency(d)
            case Operations.BACKWARD:
                condition = lambda op: op.block_id == operation.block_id and op.mb_id == operation.mb_id and \
                      op.op in [Operations.RECV_BACKWARD, Operations.FORWARD]
                deps = [op for op in schedule if condition(op)]
                for d in deps:
                    operation.add_dependency(d)
            # case Operations.RECV_BACKWARD:
            #     send_backward_op = next((op for op in schedule if op.block_id == operation.block_id + 1 and op.op == Operations.SEND_BACKWARD and op.mb_id == operation.mb_id), None)
            #     if send_backward_op:
            #         operation.add_dependency(send_backward_op)
            case Operations.SEND_FORWARD:
                deps = [op for op in schedule if op.mb_id == operation.mb_id and (\
                        (op.op == Operations.FORWARD and op.block_id == operation.block_id) or \
                            (op.op == Operations.RECV_FORWARD and op.block_id == operation.block_id + 1))]
                for d in deps:
                    operation.add_dependency(d)
            case Operations.FORWARD:
                recv_forward_op = next((op for op in schedule if op.block_id == operation.block_id and op.op == Operations.RECV_FORWARD and op.mb_id == operation.mb_id), None)
                if recv_forward_op:
                    operation.add_dependency(recv_forward_op)
            # case Operations.RECV_FORWARD:
            #     send_forward_op = next((op for op in schedule if op.block_id == operation.block_id - 1 and op.op == Operations.SEND_FORWARD and op.mb_id == operation.mb_id), None)
            #     if send_forward_op:
            #         operation.add_dependency(send_forward_op)        

    # Then, add the schedule dependencies from communications
    ## WE NEED THE PLACEMENT HERE as a comm depends on the last comm on the same DEVICE, not BLOCK
    for i in range(len(schedule)):
        current_op = schedule[i]
        if current_op.op in [Operations.FORWARD, Operations.BACKWARD]:
            continue
        for j in reversed(range(i)):
            next_op = schedule[j]
            if next_op in [Operations.FORWARD, Operations.BACKWARD] or \
                next_op.block_id != current_op.block_id: # should be rank and not block_id
                continue
            current_op.add_dependency(next_op)
            break

    return schedule[-1] # root of the entire graph

def find_cycles(graph):
    def dfs(node, visited, stack):
        visited[node] = True
        stack[node] = True

        for neighbor in node.dependencies:
            if neighbor not in visited: visited[neighbor] = False
            if not visited[neighbor]:
                if path := dfs(neighbor, visited, stack):
                    if len(path) > 1 and path[0] == path[-1]: return path
                    else: return path + [neighbor]
            elif stack[neighbor]:
                return [neighbor]

        stack[node] = False
        return False

    visited = {graph: False}
    stack = {graph: False}

    return dfs(graph, visited, stack)

def reorder_operations(operations):
    # Define the target and source operations
    target_ops = {Operations.FORWARD, Operations.BACKWARD}
    source_ops = {Operations.RECV_FORWARD, Operations.RECV_BACKWARD}
    
    # We need to iterate while keeping track of indexes because we'll modify the list
    i = 0
    while i < len(operations):
        current_op = operations[i]
        if current_op.op in target_ops:
            # Look for the next source operation
            for j in range(i + 1, len(operations)):
                next_op = operations[j]
                if next_op.op in source_ops:
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
    n_stages = len(placement)
    n_devices = max(placement) + 1

    # All forward
    for rank in range(n_devices):
        ids = [i for i in range(len(placement)) if placement[i] == rank]
        for i in range(n_micro_batches):
            for id_ in ids:
                schedule.append(Operation(id_, i, Operations.RECV_FORWARD, *options))
                schedule.append(Operation(id_, i, Operations.FORWARD, *options))
                schedule.append(Operation(id_, i, Operations.SEND_FORWARD, *options))
        
        # All backward
        for i in range(n_micro_batches):
            for id_ in reversed(ids):
                schedule.append(Operation(id_, i, Operations.RECV_BACKWARD, *options))
                schedule.append(Operation(id_, i, Operations.BACKWARD, *options))
                schedule.append(Operation(id_, i, Operations.SEND_BACKWARD, *options))
    
    assert len(schedule) == n_micro_batches * n_stages * 2 * 3
    if cycle := find_cycles(graph_from_schedule(schedule)):
        logger.warning(f'Found potential deadlocks in the schedule ! Cycle : {cycle}')
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
    # rank = int(os.getenv("RANK"))


        fwds = [0] * stages_per_device
        bwds = [0] * stages_per_device

        i = 0
        b_f = 0
        # Warmup phase : each device can compute until the micro batch forward is finished (n_stages), but it can only start after it was forwarded through all the previous layers (rank)
        while i < (stages_per_device * n_micro_batches) and i < (n_stages - rank):
            i += 1
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], Operations.RECV_FORWARD))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], Operations.FORWARD))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], Operations.SEND_FORWARD))
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
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], Operations.RECV_BACKWARD))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], Operations.BACKWARD))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], Operations.SEND_BACKWARD))
            bwds[b_b] += 1

            # Same as before, except that we can compute 2x less micro batches because half of the time is spent doing forwards
            if (i - state) % (n_devices // 2) == 0 or (i - state) % n_micro_batches == 0:
                b_b = (b_b - 1) % stages_per_device

            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], Operations.RECV_FORWARD))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], Operations.FORWARD))
            schedule.append(Operation(b_f * n_devices + rank, fwds[b_f], Operations.SEND_FORWARD))
            fwds[b_f] += 1

            if (i >= n_stages and i % (n_devices // 2) == 0) or (i % n_micro_batches) == 0:
                b_f = (b_f + 1) % stages_per_device

        while i < (stages_per_device * n_micro_batches * 2 - (stages_per_device * n_micro_batches - state)):
            i += 1
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], Operations.RECV_BACKWARD))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], Operations.BACKWARD))
            schedule.append(Operation(b_b * n_devices + rank, bwds[b_b], Operations.SEND_BACKWARD))
            bwds[b_b] += 1

            # Finish all backwards
            if (i - n_micro_batches - state) % (n_devices // 2) == 0 or (i - n_micro_batches - state) % n_micro_batches == 0:
                b_b = (b_b - 1) % stages_per_device

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
    placement = torch.tensor([0, 1, 2, 3])
    schedule = generate_1f1b_schedule(placement, 16, prefetching = True)
    if cycle := find_cycles(graph_from_schedule(schedule)):
        logger.warning(f'Found potential deadlocks in the schedule !')
        print(cycle)
    print(f'Rank {os.getenv("RANK")} - {schedule}')
