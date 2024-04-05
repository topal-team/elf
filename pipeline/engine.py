from enum import Enum
import torch.distributed as dist
from .pipeline import compute_loss

import logging
logger = logging.getLogger("engine")

class Operations(Enum):
    FORWARD = 1
    BACKWARD = 2
    SEND_FORWARD = 3
    SEND_BACKWARD = 4
    RECV_FORWARD = 5
    RECV_BACKWARD = 6
    
    def __repr__(self) -> str:
        return self.name

class StageScheduler():
    '''
    Schedule is a list of tuples (block id, operation)
    It is supposed to be feasible and correct
    '''
    def __init__(self, schedule, blocks):
        self.schedule = schedule
        self.blocks = blocks
        self.rank = self.blocks[0].rank if blocks else None
        for b in self.blocks: assert b.rank == self.rank, "All blocks in a stage should be on the same rank"
        self.id_to_block = {str(b.id): b for b in self.blocks}

    def train_step(self, batch, target, loss_fn, split_size = 1):
        '''
        Perform forward + backward pass on a batch of data
        '''
        for b in self.blocks:
            b.model.zero_grad()
        
        splits = iter(batch.split(split_size, dim=0))

        result = []
        i = 0 # TODO: change that ! maybe they're not computed in the same order
        # Add a micro_batch_id to the schedule nodes ?

        for (id_, op) in self.schedule:
            if str(id_) in self.id_to_block:
                block = self.id_to_block[str(id_)]
                logger.debug(f'Computing {op} on block {block}')
                match op:
                    case Operations.FORWARD:
                        y = block.forward()
                        if y is not None: result.append(y)
                    case Operations.BACKWARD:
                        block.backward()
                    case Operations.SEND_FORWARD:
                        block.send_forward()
                    case Operations.SEND_BACKWARD:
                        block.send_backward()
                    case Operations.RECV_FORWARD:
                        if block.previous is None:
                            block.inputs.append((None, next(splits)))
                        block.recv_forward()
                    case Operations.RECV_BACKWARD:
                        if block.next is None:
                            compute_loss(block, result[i], target[i*split_size:(i + 1) * split_size], loss_fn)
                            i += 1
                        block.recv_backward()
                    case _:
                        raise Exception(f'Unknown operation : {op}')

        logger.debug(f'[Rank {self.rank}] - Finished computation !')
        dist.barrier()
        return result
