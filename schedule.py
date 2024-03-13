import os
from pipeline import compute_loss

import torch.distributed as dist

DEBUG = "DEBUG" in os.environ and os.environ["DEBUG"] != "0"

def train_step_afab(blocks, batch, target, loss_fn):
    splits = iter(batch.split(1, dim=0))
    # Schedule here !
    # All forward
    result = []
    for b in blocks:
        for i in range(batch.size(0)):
            # Feed micro-batch in pipeline
            if b.previous is None:
                if DEBUG: print(f'{b} - Feeding micro batch {i} into the pipeline')
                b.inputs.append(next(splits))

            b.recv_forward()
            b.forward()
            b.send_forward()
            # Last layer has the result
            if b.next is None: result.append(b.act_to_send.popleft())

    grads = []
    # All backward
    for b in reversed(blocks):
        for i in range(batch.size(0)):
            # Compute loss on last device
            if b.next is None:
                if DEBUG: print(f'{b} - Starting backward pass for micro batch {i}')
                compute_loss(b, result[i], target[i], loss_fn)

            b.recv_backward()
            b.backward()
            b.send_backward()
            if b.previous is None: grads.append(b.grads_to_send.popleft())

    return result, grads
