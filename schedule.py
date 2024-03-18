import os
from pipeline import compute_loss

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
            if b.next is None: result.append(b.act_to_send.popleft().unsqueeze(0))

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

def train_step_1f1b(blocks, batch, target, loss_fn, n_stages):
    '''
    Not interleaved for now !!
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
                block.forward()
                block.send_forward()
                if block.next is None: result.append(block.act_to_send.popleft())

            bwd_offset = n_stages - 1 - block.id
            # Warmup phase
            if i >= bwd_offset:
                # One backward
                if block.next is None:
                    compute_loss(block, result[i - bwd_offset], target[i - bwd_offset], loss_fn)

                block.recv_backward()
                block.backward()
                block.send_backward()
                if block.previous is None: grads.append(block.grads_to_send.popleft())
    
    return result, grads
