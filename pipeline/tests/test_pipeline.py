import torch
import torch.nn as nn
import torch.distributed as dist
from ..pipeline import *
import pytest

@pytest.fixture
def init_dist():
    assert "RANK" in os.environ, "Cannot run multi-process tests without torchrun !"

    rank = int(os.getenv("RANK"))
    local_rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend = "nccl")
    yield rank, local_rank, world_size
    if dist.is_initialized():
        dist.destroy_process_group()

@pytest.mark.single
def test_metadata():
    # Test creation from class method from_tensor
    received = torch.tensor([dtypes.index(torch.float32), 2, 3, 4])
    metadata = TensorMetadata.from_tensor(received)

    assert metadata.dtype == torch.float32
    assert metadata.shape == [2, 3, 4]

    # Test initialization by constructor
    t = torch.empty((6, 3, 2), dtype = torch.bfloat16)
    metadata = TensorMetadata(t)
    assert metadata.dtype == torch.bfloat16
    assert metadata.shape == torch.Size([6, 3, 2])

    # Test tensor representation of metadata
    t = torch.empty((3, 1, 4), dtype = torch.float16)
    metadata = TensorMetadata(t)
    tensor_repr = metadata.to_tensor()
    assert tensor_repr[0] == dtypes.index(torch.float16)
    assert tensor_repr[1] == 3
    assert tensor_repr[2] == 1
    assert tensor_repr[3] == 4
    assert tensor_repr[4] == 0

    # Test buffer creation
    t = torch.empty((8, 5, 2), dtype = torch.int64)
    metadata = TensorMetadata(t)
    buffer = metadata.get_buffer()
    assert buffer.dtype == torch.int64
    assert buffer.shape == torch.Size([8, 5, 2])

    return

class FakeWorker():
    def __init__(self, done):
        self.done = done

    def wait(self):
        while not self.done: pass
        return

    def is_completed(self):
        return self.done

@pytest.mark.single
def test_block():
    '''
    TODO: Tests for communications (send/recv) ; probably need multiple processes
    '''
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    model = nn.Linear(2, 1, bias = False)

    # Test pipe links
    block = PipelineBlock(model, id_ = 0, placement = [0, 2])

    assert block.id == 0
    assert block.previous is None
    assert block.next == 2
    assert block.rank == 0

    block = PipelineBlock(model, id_ = 1, placement = [1, 2])
    assert block.id == 1
    assert block.previous == 1
    assert block.next is None
    assert block.rank == 2

    block = PipelineBlock(model, id_ = 1, placement = [2, 1, 0])
    assert block.id == 1
    assert block.previous == 2
    assert block.next == 0
    assert block.rank == 1

    # Test forward pass
    block.model.weight = nn.Parameter(torch.tensor([3., -1.], device = device))
    assert len(block.inputs) == 0
    inputs = torch.tensor([2., 4.], device = device)
    block.inputs.append((FakeWorker(True), inputs))

    block.forward()

    expected_result = torch.tensor([2.], device = device)
    assert torch.allclose(block.activations[0], expected_result)
    assert torch.allclose(block.act_to_send[0], expected_result)
    assert len(block.inputs) == 0
    assert len(block.inputs_to_keep) == 1

    # Test backward pass
    block.grads.append((FakeWorker(True), torch.tensor([3.], device = device)))

    block.backward()
    expected_grads_weights = torch.tensor([6., 12.], device = device)
    expected_grads_inputs = torch.tensor([9., -3.], device = device)

    assert torch.allclose(block.model.weight.grad.data, expected_grads_weights)
    assert torch.allclose(inputs.grad.data, expected_grads_inputs)

    assert len(block.activations) == 0
    assert len(block.inputs_to_keep) == 0
    assert len(block.grads_to_send) == 1

@pytest.mark.single
def test_pipeline():
    '''
    TODO: Test full forward / backward pass ; probably needs multiple processes :)
    '''

    os.environ["WORLD_SIZE"] = "3"

    # Test automatic placement + partitioning
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4), nn.Linear(4, 5))
    try:
        _ = Pipeline(model)
        assert torch.cuda.device_count() == len(model), f'The partitioning should require {os.getenv("WORLD_SIZE")} devices to run'
    except RuntimeError:
        pass

    # Test predefined placement
    pipe = Pipeline(model, placement = ['cpu', 'cpu'])
    assert len(pipe.blocks) == 2

    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        os.environ["RANK"] = "1"
        pipe = Pipeline(model, placement = ['cpu', 'cpu', '1'])
        assert len(pipe.blocks) == 1

    # Test predefined partition
    pipe = Pipeline([l for l in model.children()], partition = None)
    assert len(pipe.blocks) == 0 # not the right rank
    assert pipe.placement == list(range(len(model))) # default value

@pytest.mark.multi
def test_block_multi(init_dist):
    rank, local_rank, world_size = init_dist

    placement = [0, 1]
    model = nn.Linear(rank + 1, rank + 2, bias = False)
    block = PipelineBlock(model, rank, placement)

    assert block.rank == rank
    if rank == 0:
        assert block.previous is None
        assert block.next == 1

        block.model.weight = nn.Parameter(torch.tensor([[2.0], [2.0]], device = local_rank))

        assert len(block.inputs) == 0
        block.recv_forward() # Should do nothing as block has no previous
        assert len(block.inputs) == 0
        block.inputs.append(torch.ones((2, 1), device = local_rank))

        assert len(block.activations) == 0
        block.forward()
        assert len(block.activations) == 1
        assert len(block.act_to_send) == 1
        assert len(block.inputs) == 0
        assert len(block.inputs_to_keep) == 1
        assert torch.allclose(block.activations[0], torch.full((2, 2), 2.0))

        block.send_forward() # Should be matched by a recv_forward
        assert len(block.act_to_send) == 0

    elif rank == 1:
        assert block.previous == 0
        assert block.next is None

        block.model.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], device = local_rank))

        assert len(block.inputs) == 0
        block.recv_forward()
        assert len(block.inputs) == 1
        assert len(block.inputs_to_keep) == 1

        assert len(block.activations) == 0
        y = block.forward()
        assert torch.allclose(y, torch.full((2, 3), 6.0, device = local_rank))

        assert len(block.act_to_send) == 0 # no next block, nothing to send
        block.send_forward() # does nothing
        assert len(block.act_to_send) == 0
