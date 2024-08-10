'''
Execution of the pipeline
'''

import torch
import torch.distributed as dist
import time
import torch.distributed as dist
from .schedule import OperationType
from .utils import Timer

import logging
logger = logging.getLogger("engine")

def op_to_str(op):
    '''
    Pretty print for dist.P2POp

    :param op: communication operation
    :type op: dist.P2POp
    :return: string describing the op
    :rtype: string
    '''
    match op.op:
        case dist.isend:
            return f'Send to {op.peer}'
        case dist.irecv:
            return f'Receive from {op.peer}'

class Engine():
    '''
    Coordinates the execution of a schedule on a list of blocks at a device/rank level.
    Takes care of feeding the input to the first block and computing the loss on the last block.
    '''
    def __init__(self, blocks):
        '''
        :param blocks: list of blocks handled by this process rank
        :type blocks: List[PipelineBlock]
        '''
        self.blocks = blocks
        self.rank = self.blocks[0].rank if blocks else None
        for b in self.blocks: assert b.rank == self.rank, "All blocks in a stage should be on the same rank"
        self.id_to_block = {str(b.id): b for b in self.blocks}
        self.comms = []

    def _run_comms(self):
        '''
        Run all currently batched communications for this device
        Internal function, this should not be used by the user
        '''
        if len(self.comms) == 0: return 0
        
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            works = dist.batch_isend_irecv(self.comms)
            logger.debug(f'Rank {self.rank} - Running batched communications {[op_to_str(c) for c in self.comms]}')
            for w in works: w.wait()
            
        self.comms.clear()
        stream.synchronize()

    def train_step(self, batch, target, loss_fn, schedule, mb_sizes, profile = False):
        '''
        Executes a schedule on a batch of data

        :param batch: input data, only used on the first block of the pipeline
        :type batch: Tensor
        :param target: groundtruth, only used on the last block of the pipeline
        :type target: Tensor
        :param loss_fn: loss function to use ; we recommend using the torch built-in function, but if you want to use your own it needs to take the same parameter "reduction = 'sum'" as torch ones.
        :type loss_fn: function (Tensor, Tensor, reduction = 'sum') -> Tensor
        :param schedule: list of operations. For more info, see schedule.
        :type schedule: list[Operation]
        :param split_size: list of micro batch sizes. The list should cover the entire batch, i.e. ``sum(split_size) == batch_size``
        :type split_size: int or List[int]
        :param profile: Whether to activate nvidia profiling or not. If True, NVTX ranges will be generated for each operation
        :type profile: boolean

        :return:

        - Result of the forward pass
        - Losses for each micro-batch
        - Insights about time taken, as a dict containing:

            - total: total time taken for the execution
            - idle: total time not used for computation for this process
            - start_idle: time between the start of execution and the first computation
            - end_idle: time between the last computation and the end of execution
            - bubble: idle time between first and last computation

        :rtype: Tensor, Tensor, Dict[float]
        '''
        split_batches = [tensor.split(mb_sizes, dim = 0) for tensor in batch]
        microbatches = iter(zip(*split_batches))

        result = []
        losses = []
        current_target = (0, 0) # micro batch id, position in target tensor

        dist.barrier() # useful for timing, but it probably slows down the execution a bit
        pipe_start = time.time()
        warmup_start = None
        
        for op in schedule:
            block = self.id_to_block.get(str(op.block_id))
            if block is None: continue # not my job

            logger.debug(f'Computing {op} on block {block} with options {op.options}')
            if profile:
                torch.cuda.nvtx.range_push(f'{block}:{op}')
            if warmup_start is None and op.op != OperationType.RECV_FORWARD:
                torch.cuda.synchronize()
                warmup_start = time.time()
            match op.op:
                case OperationType.FORWARD:
                    self._run_comms()
                    y = block.forward(**op.options)
                    if y is not None:
                        result.append(*y.values())
                        
                case OperationType.BACKWARD:
                    if block.next is None:
                        i, start = current_target
                        end = start + mb_sizes[i]
                        with Timer() as timer:
                            loss = compute_loss(block, result[i], target[start:end], loss_fn)
                        block.compute_time += timer.time()
                        losses.append(loss)
                        logger.debug(f'{block} - Computed loss = {loss}')
                        current_target = (i + 1, end)

                    self._run_comms()
                    block.backward(**op.options)
                    
                case OperationType.SEND_FORWARD:
                    if (op.options.get("dst") or block.next) == block.rank:
                        # The next block is on the same device ; we want to bypass p2p comms
                        next_block = self.id_to_block.get(str(op.block_id + 1))
                        next_block.inputs.append({key: [None, value.detach()] for key, value in block.act_to_send.popleft().items()})

                    if comm := block.send_forward(**op.options):
                        self.comms.extend(comm)
                        
                case OperationType.SEND_BACKWARD:
                    if (op.options.get("src") or block.previous) == block.rank:
                        # The previous block is on the same device ; we want to bypass p2p comms
                        next_block = self.id_to_block.get(str(op.block_id - 1))
                        next_block.grads_to_backward.append({key: [None, value.detach()] for key, value in block.grads_to_send.popleft().items()})

                    if comm := block.send_backward(**op.options):
                        self.comms.extend(comm)
                        
                case OperationType.RECV_FORWARD:
                    if block.previous is None:
                        microbatch = next(microbatches)
                        block.inputs_to_forward.append({key: [None, value] for key, value in zip(block.inputs, microbatch)})

                    if comm := block.recv_forward(mb_sizes[op.mb_id], **op.options):
                        self.comms.extend(comm)
                        
                case OperationType.RECV_BACKWARD:
                    if comm := block.recv_backward(mb_sizes[op.mb_id], **op.options):
                        self.comms.extend(comm)
                        
                case _:
                    raise Exception(f'Unknown operation : {op}')
        
            if profile:
                torch.cuda.nvtx.range_pop()

        logger.debug(f'[Rank {self.rank}] - Finished execution !')
        
        self._run_comms()
        torch.cuda.synchronize()
        cooldown_end = time.time()
        dist.barrier()
        pipe_end = time.time()
        compute_time = 0
        for block in self.blocks:
            compute_time += block.compute_time
            block.compute_time = 0

        times = {
            "total": pipe_end - pipe_start,
            "idle": pipe_end - pipe_start - compute_time,
            "start_idle": warmup_start - pipe_start,
            "end_idle": pipe_end - cooldown_end
        }
        times["bubble"] = times["idle"] - times["start_idle"] - times["end_idle"]
        return result, losses, times

def compute_loss(block, output, target, loss_fn):
    '''
    Computes the loss and correctly prepares the gradients for the pipelined backward pass

    :param block: last block of the pipeline
    :type block: PipelineBlock
    :param output: output of this block
    :type output: Tensor
    :param target: target value
    :type target: Tensor
    :param loss_fn: loss function to compute
    :type loss_fn: function (Tensor, Tensor, reduction = 'sum') -> Tensor

    :return: loss value
    :rtype: Tensor
    '''
    output = output.detach()
    output.requires_grad = True
    loss = loss_fn(output, target, reduction = "sum")
    loss.backward()
    key = list(block.outputs)[0] # TODO: multiple outputs
    block.grads_to_backward.append({key: [None, output.grad.data]})
    return loss
