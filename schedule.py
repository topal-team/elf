from pipeline import compute_loss
from engine import Operations

import logging
logger = logging.getLogger("schedule")

def train_step_afab(blocks, batch, target, loss_fn):
    '''
    All Forward All Backward schedule from GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
    https://arxiv.org/abs/1811.06965
    '''
    splits = iter(batch.split(1, dim=0))
    # Schedule here !
    # All forward
    result = []
    for b in blocks:
        for i in range(batch.size(0)):
            # Feed micro-batch into the pipeline
            if b.previous is None:
                logger.debug(f'{b} - Feeding micro batch {i} into the pipeline')
                b.inputs.append(next(splits))

            b.recv_forward()
            y = b.forward()
            b.send_forward()
            if y is not None: result.append(y)

    grads = []
    # All backward
    for b in reversed(blocks):
        for i in range(batch.size(0)):
            # Compute loss on last device
            if b.next is None:
                logger.debug(f'{b} - Starting backward pass for micro batch {i}')
                compute_loss(b, result[i], target[i], loss_fn)

            b.recv_backward()
            g = b.backward()
            b.send_backward()
            if g is not None: grads.append(g)

    return result, grads

def train_step_1f1b(blocks, batch, target, loss_fn, n_stages):
    '''
    One Forward One Backward schedule from PipeDream: Fast and Efficient Pipeline Parallel DNN Training
    https://arxiv.org/abs/1806.03377
    !! Does not support interleaving for now !!
    '''
    splits = iter(batch.split(1, dim=0))
    result = []
    grads = []

    for block in blocks:
        for i in range(batch.size(0) + n_stages - 1 - block.id):
            if i < batch.size(0): # There are still forwards left
                # One forward
                if block.previous is None:
                    block.inputs.append(next(splits))

                block.recv_forward()
                y = block.forward()
                block.send_forward()
                if y is not None: result.append(y)

            bwd_offset = n_stages - 1 - block.id
            # Warmup phase
            if i >= bwd_offset:
                # One backward
                if block.next is None:
                    compute_loss(block, result[i - bwd_offset], target[i - bwd_offset], loss_fn)

                block.recv_backward()
                g = block.backward()
                block.send_backward()
                if g is not None: grads.append(g)
    
    return result, grads



def generate_afab_schedule(placement, n_micro_batches):
    schedule = []
    # All forward
    for _ in range(n_micro_batches):
        for id_ in range(len(placement)):
            schedule.append((id_, Operations.RECV_FORWARD))
            schedule.append((id_, Operations.FORWARD))
            schedule.append((id_, Operations.SEND_FORWARD))
    
    # All backward
    for _ in range(n_micro_batches):
        for id_ in reversed(range(len(placement))):
            schedule.append((id_, Operations.RECV_BACKWARD))
            schedule.append((id_, Operations.BACKWARD))
            schedule.append((id_, Operations.SEND_BACKWARD))
    
    assert len(schedule) == n_micro_batches * len(placement) * 2 * 3

    return schedule

def generate_1f1b_schedule(placement, n_micro_batches):
    schedule = []

    for _ in range(n_micro_batches):
        for id_ in range(len(placement)):
            # schedule.append((id_, Operations.RECV_FORWARD))
            schedule.append((id_, Operations.FORWARD))
            # schedule.append((id_, Operations.SEND_FORWARD))

    # Interleave with backwards
    i = len(placement) * 1
    b = len(placement) - 1
    offset = len(placement) - 1
    for _ in range(n_micro_batches):
        for b in reversed(range(len(placement))):
            # schedule.insert(i, (b, Operations.RECV_BACKWARD))
            schedule.insert(i, (b, Operations.BACKWARD))
            # schedule.insert(i + 2, (b, Operations.SEND_BACKWARD))

            i += offset
            offset = 1 if offset == 1 else offset - (1 * 1)
            if i < len(schedule): i+= 1

    assert len(schedule) == n_micro_batches * len(placement) * 2 * 1

    return schedule

if __name__ == "__main__":
    import torch
    placement = torch.tensor([0, 1, 2, 3])
    schedule = generate_1f1b_schedule(placement, 4)
    for rank in range(placement.max().item() + 1):
        actions = [(id_, op) for id_, op in schedule if id_ == rank]
        print(f'Rank {rank} - {actions}')
