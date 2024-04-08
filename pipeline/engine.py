from enum import Enum
import torch.distributed as dist
from pipeline import compute_loss
import time
import os
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
        self.rank = self.blocks[0].rank.item() if blocks else None
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
        f = open(viz_file + str(self.rank), 'w') if viz_file is not None else None
        
        for (id_, op) in self.schedule:
            if str(id_) in self.id_to_block:
                block = self.id_to_block[str(id_)]
                logger.debug(f'Computing {op} on block {block}')
                if f is not None: f.write(f'{self.rank}:{time.time()}:{id_},{op}\n')
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
        if f is not None:
            f.close()
            if self.rank == 0:
                with open(viz_file, 'w') as outfile: [outfile.write(open(f, 'r').read()) for f in [viz_file + str(r) for r in range(int(os.environ["WORLD_SIZE"]))]]
            dist.barrier()    
            os.remove(viz_file + str(self.rank))
        return result

def visualize(path):
    operations = {}  # Dictionary to store operations by device
    with open(path, 'r') as file:
        for line in file:
            rank, times, id_op = line.split(':')
            times = float(times)  # Convert time to float
            
            if rank not in operations:
                operations[rank] = []
            operations[rank].append((times, id_op.strip()))

    colors = {
        str(Operations.RECV_FORWARD): "gold",
        str(Operations.FORWARD): "springgreen",
        str(Operations.SEND_FORWARD): "orange",
        str(Operations.RECV_BACKWARD): "plum",
        str(Operations.BACKWARD): "red",
        str(Operations.SEND_BACKWARD): "fuchsia",
    }


    fig, ax = plt.subplots()
    for rank, ops in operations.items():
        # Sort operations by start time to ensure correct sequential plotting
        ops.sort(key=lambda x: x[0])
        
        for i, (start_time, id_op) in enumerate(ops):
            if i + 1 < len(ops):
                duration = ops[i + 1][0] - start_time
            else:
                duration = 0.1  # Default duration for the last operation

            # Plot each operation as a horizontal bar
            bar = ax.barh(y=int(rank), width=duration, left=start_time, height=0.8, color=colors[id_op.split(',')[1]], edgecolor='black')
            
            # Annotate the bar with the operation ID
            # Adjust the position to be inside the bar, slightly to the right and vertically centered
            text_x = start_time + 0.2 * duration  # Adjust this value as needed
            text_y = int(rank)
            # print(f'{duration}, {(ops[-1][0] - ops[0][0]) / 10}')
            if duration > (ops[-1][0] - ops[0][0]) / 10:
                ax.text(text_x, text_y, id_op, va='center', ha='left', fontsize=16, color='black')

    ax.set_xlabel('Time')
    ax.set_ylabel('Device Rank')
    plt.show()

if __name__ == "__main__":
    import sys
    path = sys.argv[1]

    visualize(path)

