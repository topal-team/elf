import torch.distributed as dist
import time
import os
import matplotlib
import matplotlib.pyplot as plt
from .schedule import OperationType

import logging
logger = logging.getLogger("engine")

def op_to_str(op):
    '''
    Pretty print for dist.P2POp
    '''
    match op.op:
        case dist.isend:
            return f'Send to {op.peer}'
        case dist.irecv:
            return f'Receive from {op.peer}'

class Engine():
    '''
    Schedule is a list of tuples (block id, operation)
    It is supposed to be feasible and correct
    '''
    def __init__(self, blocks):
        self.blocks = blocks
        self.rank = self.blocks[0].rank if blocks else None
        for b in self.blocks: assert b.rank == self.rank, "All blocks in a stage should be on the same rank"
        self.id_to_block = {str(b.id): b for b in self.blocks}
        self.comms = []

    def _run_comms(self):
        if len(self.comms) == 0: return
        works = dist.batch_isend_irecv(self.comms)
        logger.debug(f'Rank {self.rank} - Running batched communications {[op_to_str(c) for c in self.comms]}')
        for w in works: w.wait()
        self.comms.clear()

    def train_step(self, batch, target, loss_fn, schedule, split_size, viz_file = None):
        '''
        Perform forward + backward pass on a batch of data
        '''
        splits = iter(batch.split(split_size, dim=0))

        result = []
        losses = []
        i = 0 # TODO: change that ! maybe they're not computed in the same order
        # Add a micro_batch_id to the schedule nodes ?
        curr = 0
        if viz_file is not None:
            stats = []
            
        for op in schedule:
            if str(op.block_id) in self.id_to_block:
                block = self.id_to_block[str(op.block_id)]
                logger.debug(f'Computing {op} on block {block}')
                if viz_file is not None: stats.append((time.time(), id_, op))
                match op.op:
                    case OperationType.FORWARD:
                        self._run_comms()
                        y = block.forward(op.options)
                        if y is not None:
                            result.append(y)
                    case OperationType.BACKWARD:
                        self._run_comms()
                        block.backward(op.options)
                    case OperationType.SEND_FORWARD:
                        if comm := block.send_forward(op.options):
                            self.comms.append(comm)
                    case OperationType.SEND_BACKWARD:
                        if comm := block.send_backward(op.options):
                            self.comms.append(comm)
                    case OperationType.RECV_FORWARD:
                        if block.previous is None:
                            block.inputs.append((None, next(splits)))
                        if comm := block.recv_forward(split_size[op.mb_id], op.options):
                            self.comms.append(comm)
                    case OperationType.RECV_BACKWARD:
                        if block.next is None:
                            nexti = curr + split_size[i]
                            loss = compute_loss(block, result[i], target[curr:nexti], loss_fn)
                            losses.append(loss)
                            logger.debug(f'{block} - Computed loss = {loss}')
                            i += 1
                            curr = nexti
                        if comm := block.recv_backward(split_size[op.mb_id], op.options):
                            self.comms.append(comm)
                    case _:
                        raise Exception(f'Unknown operation : {op}')
        
        logger.debug(f'[Rank {self.rank}] - Finished computation !')
        if viz_file is not None: stats.append((time.time(), self.rank, ".END"))
        self._run_comms()
        dist.barrier()
        if viz_file is not None:
            for r in range(int(os.environ["WORLD_SIZE"])):
                # Each process writes in turn
                if r == self.rank:
                    if r == 0: open(viz_file, "w").close() # erase content
                    with open(viz_file, 'a') as outfile:
                        for t, id_, op in stats:
                            outfile.write(f'{self.rank}:{t}:{id_},{op}\n')

                dist.barrier()
        return result, losses

def compute_loss(block, output, target, loss_fn):
    '''
    Computes the loss and correctly prepares the gradients for the pipelined backward pass
    '''
    output = output.detach()
    output.requires_grad = True
    loss = loss_fn(output, target, reduction = "sum")
    loss.backward()
    block.grads.append((None, output.grad.data))
    return loss

def visualize(path):
    '''
    Display the execution time of each operation executed by the engine
    path: path to a file that should be created by a Pipeline/Engine call
    '''
    operations = {}  # Dictionary to store operations by device
    with open(path, 'r') as file:
        for line in file:
            rank, times, id_op = line.split(':')
            times = float(times)  # Convert time to float
            
            if rank not in operations:
                operations[rank] = []
            operations[rank].append((times, id_op.strip()))

    colors = {
        str(OperationType.RECV_FORWARD): "gold",
        str(OperationType.FORWARD): "springgreen",
        str(OperationType.SEND_FORWARD): "orange",
        str(OperationType.RECV_BACKWARD): "plum",
        str(OperationType.BACKWARD): "red",
        str(OperationType.SEND_BACKWARD): "fuchsia",
        ".END": "black"
    }

    fig, ax = plt.subplots()
    first_start = 0
    for rank, ops in operations.items():
        # Sort operations by start time to ensure correct sequential plotting
        ops.sort(key=lambda x: x[0])
        if ops[0][0] < first_start or first_start == 0:
            first_start = ops[0][0]

    for rank, ops in operations.items():
        # Sort operations by start time to ensure correct sequential plotting
        if ops[0][0] < first_start or first_start == 0:
            first_start = ops[0][0]
        
        for i, (start_time, id_op) in enumerate(ops):
            if i + 1 < len(ops):
                duration = ops[i + 1][0] - start_time
            else:
                duration = 0

            # Plot each operation as a horizontal bar, *1000 for ms instead of s 
            ax.barh(y=int(rank), width=duration * 1000, left=(start_time - first_start) * 1000, height=0.8, color=colors[id_op.split(',')[1]], edgecolor='black')
            
            # Annotate the bar with the operation ID
            # Adjust the position to be inside the bar, slightly to the right and vertically centered
            # text_x = start_time + 0.15 * duration  # Adjust this value as needed
            # text_y = int(rank)
            # if duration > (ops[-1][0] - ops[0][0]) / 12:
            #     ax.text(text_x, text_y, id_op, va='center', ha='left', fontsize=12, color='black')


    legend_handles = [matplotlib.patches.Patch(facecolor=color, edgecolor='black', label=label.split('.')[1].lower()) for label, color in colors.items() if label != ".END"]
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Device Rank')
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)) # show only integers
    plt.show()

if __name__ == "__main__":
    import sys
    path = sys.argv[1]

    visualize(path)
