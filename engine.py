from enum import Enum
from pipeline import compute_loss

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
        self.rank = self.blocks[0].rank
        for b in self.blocks: assert b.rank == self.rank, "All blocks in a stage should be on the same rank"
        self.id_to_block = {str(b.id): b for b in self.blocks}

    def train_step(self, batch, target, loss_fn):
        last_block = max(map(lambda s: s[0], self.schedule))
        
        splits = iter(batch.split(1, dim=0)) # adjust micro batch size

        result = []
        grads = []

        for (id, op) in self.schedule:
            if str(id) in self.id_to_block:
                block = self.id_to_block[str(id)]
                match op:
                    case Operations.FORWARD:
                        y = block.forward()
                        if y is not None: result.append(y)
                    case Operations.BACKWARD:
                        g = block.backward()
                        if g is not None: grads.append(g)
                    case Operations.SEND_FORWARD:
                        block.send_forward()
                    case Operations.SEND_BACKWARD:
                        block.send_backward()
                    case Operations.RECV_FORWARD:
                        if id == 0:
                            block.inputs.append(next(splits)) # When merging with async, don't forget to add (None, )
                        block.recv_forward()
                    case Operations.RECV_BACKWARD:
                        if id == last_block:
                            compute_loss(block, result[-1], target[len(result) - 1], loss_fn)
                        block.recv_backward()
                    case _:
                        raise f'Unknown operation : {op}'

        return result, grads