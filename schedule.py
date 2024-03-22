from engine import Operations

import logging
logger = logging.getLogger("schedule")

def generate_afab_schedule(placement, n_micro_batches):
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
    schedule = []
    n_stages = len(placement)
    
    # Warmup phase
    for b in range(n_stages):
        # End of warmup phase is either first backward or end of batch
        warmup_fwd = min(n_micro_batches - 1, 2 * (n_stages - b - 1))
        for i in range(warmup_fwd):
            schedule.append((b, Operations.FORWARD))

    # Steady phase
    for b in range(n_stages):
        remaining_fwd = n_micro_batches - min(n_micro_batches - 1, 2 * (n_stages - b - 1))
        for i in range(remaining_fwd):
            schedule.append((b, Operations.FORWARD))
            schedule.append((b, Operations.BACKWARD))

    # Cooldown phase
    for b in reversed(range(n_stages)):
        remaining_bwd = min(n_micro_batches - 1, 2 * (n_stages - b - 1))
        for i in range(remaining_bwd):
            schedule.append((b, Operations.BACKWARD))

    assert len(schedule) == n_micro_batches * n_stages * 2 * 1

    return schedule

if __name__ == "__main__":
    import torch
    placement = torch.tensor([0, 1, 2, 3])
    schedule = generate_1f1b_schedule(placement, 8)
    for rank in range(placement.max().item() + 1):
        actions = [(id_, op) for id_, op in schedule if id_ == rank]
        print(f'Rank {rank} - {actions}')

    print(schedule)
