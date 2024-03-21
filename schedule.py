import logging
from pipeline import compute_loss
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
