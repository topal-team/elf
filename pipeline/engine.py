import torch
import torch.distributed as dist
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

    def train_step(self, batch, target, loss_fn, schedule, split_size, profile = None):
        '''
        Perform forward + backward pass on a batch of data
        '''
        splits = iter(batch.split(split_size, dim=0))

        result = []
        losses = []
        i = 0 # TODO: change that ! maybe they're not computed in the same order
        # Add a micro_batch_id to the schedule nodes ?
        curr = 0
            
        for op in schedule:
            if str(op.block_id) in self.id_to_block:
                block = self.id_to_block[str(op.block_id)]
                logger.debug(f'Computing {op} on block {block}')
                if profile is not None:
                    torch.cuda.nvtx.range_push(f'{block}:{op}')
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
        
        if profile is not None:
            torch.cuda.nvtx.range_pop()
        logger.debug(f'[Rank {self.rank}] - Finished computation !')
        self._run_comms()
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
