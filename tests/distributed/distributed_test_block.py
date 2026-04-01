import torch
import torch.nn as nn
import torch.distributed as dist

from elf.execution.block import PipelineBlock
from elf.partitioners.utils import Signature

from tests.distributed.distributed_utils import test


def test_block_communication(rank, local_rank, world_size):
	if world_size < 2:
		print("This test needs at least 2 processes to run")
		return

	if rank == 0:
		model = nn.Linear(2, 2)
		model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=local_rank)
		model.bias.data = torch.tensor([0.1, 0.2], device=local_rank)

		signature = Signature(inputs=["input"], outputs=["output"], sources=[None], targets=[[1]])

		block = PipelineBlock(
			model=model, id_=0, placement=[0, 1], signature=signature, pp_group=None, dp_group=None
		)

		input_tensor = torch.tensor([[1.0, 2.0]], device=local_rank)
		block.input_variables[0].set(block.input_variables[0].to_process, 0, (None, input_tensor))
		block.forward(0)
		block.send_forward(0, dst=1)

		block.recv_backward(0, mb_size=1, src=1)
		block.backward_inputs(0)
		block.backward_params(0)

	elif rank == 1:
		model = nn.Linear(2, 2)
		model.weight.data = torch.tensor([[0.5, 0.6], [0.7, 0.8]], device=local_rank)
		model.bias.data = torch.tensor([0.3, 0.4], device=local_rank)

		signature = Signature(inputs=["input"], outputs=["output"], sources=[0], targets=[[None]])

		block = PipelineBlock(
			model=model, id_=1, placement=[0, 1], signature=signature, pp_group=None, dp_group=None
		)

		block.recv_forward(0, mb_size=1, src=0)
		block.forward(0)

		grad_tensor = torch.tensor([[1.0, 1.0]], device=local_rank)
		block.output_variables[0][0].set(
			block.output_variables[0][0].to_process, 0, (None, grad_tensor)
		)
		block.backward_inputs(0)
		block.backward_params(0)
		block.send_backward(0, dst=0)

	dist.barrier()


def main():
	test(test_block_communication)


if __name__ == "__main__":
	main()
