from enum import Enum
import torch.distributed as dist
from pipeline import compute_loss
import time
import matplotlib.pyplot as plt

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

    def train_step(self, batch, target, loss_fn, split_size = 1, viz_file = None):
        '''
        Perform forward + backward pass on a batch of data
        '''        
        splits = iter(batch.split(split_size, dim=0))

        result = []
        i = 0 # TODO: change that ! maybe they're not computed in the same order
        # Add a micro_batch_id to the schedule nodes ?
        f = open(viz_file, 'w') if viz_file is not None else None

        for (id_, op) in self.schedule:
            if str(id_) in self.id_to_block:
                block = self.id_to_block[str(id_)]
                logger.debug(f'Computing {op} on block {block}')
                if f is not None: f.write(f'{self.rank}:{time.time()}:{id_},{op}')
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


        # f.close()
        logger.debug(f'[Rank {self.rank}] - Finished computation !')
        dist.barrier()
        return result

    # ChatGPT generated :)    
    def visualize(self, path):
        operations = {}  # Dictionary to store operations by device
        with open(path, 'r') as file:
            for line in file:
                rank, time_id_op = line.split(':')
                time, id_op = time_id_op.split(',')
                time = float(time)  # Convert time to float
                
                if rank not in operations:
                    operations[rank] = []
                operations[rank].append((time, id_op.strip()))

        # Function to plot the operations
        fig, ax = plt.subplots()
        for rank, ops in operations.items():
            times = [op[0] for op in ops]  # Extract times
            ids = [int(rank)] * len(ops)  # Use device rank as y-value
            
            ax.scatter(times, ids, label=f'Device {rank}', s=100)  # s is the marker size
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Device Rank')
        ax.legend()
        plt.show()