import torch
import torch.nn as nn

import torch.distributed as dist
from ..pipeline import *
from ..utils import dtypes
import pytest

@pytest.fixture
def init_dist():
    assert "RANK" in os.environ, "Cannot run multi-process tests without torchrun !"

    rank = int(os.getenv("RANK"))
    local_rank = int(os.getenv("LOCAL_RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    torch.cuda.set_device(local_rank)
    try:
        if not dist.is_initialized():
            dist.init_process_group(backend = "nccl")
        
        yield rank, local_rank, world_size
    finally:
        if dist.is_initialized():
            dist.barrier()
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
    buffer = metadata.get_buffer(3)
    assert buffer.dtype == torch.int64
    assert buffer.shape == torch.Size([3, 8, 5, 2])

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
    block.grads.append((FakeWorker(True), torch.tensor(3., device = device)))

    block.backward()
    expected_grads_weights = torch.tensor([6., 12.], device = device)
    expected_grads_inputs = torch.tensor([9., -3.], device = device)

    assert torch.allclose(block.model.weight.grad.data, expected_grads_weights)
    assert torch.allclose(inputs.grad.data, expected_grads_inputs)

    assert len(block.activations) == 0
    assert len(block.inputs_to_keep) == 0
    assert len(block.grads_to_send) == 1

@pytest.mark.multi
def test_block_multi(init_dist):
    rank, local_rank, world_size = init_dist
    
    if rank > 1:
        pytest.skip("This test only needs 2 processes")
        return

    placement = [0, 1]
    model = nn.Linear(rank + 1, rank + 2, bias = False)
    block = PipelineBlock(model, rank, placement)
    block.metadata = TensorMetadata(torch.ones((rank + 1,), device = local_rank))
    block.out_metadata = TensorMetadata(torch.ones((rank + 2,), device = local_rank))
    
    assert block.rank == rank
    if rank == 0:
        assert block.previous is None
        assert block.next == 1

        block.model.weight = nn.Parameter(torch.tensor([[2.0], [2.0]], device = local_rank))

        assert len(block.inputs) == 0
        w = block.recv_forward(1) # Should do nothing as block has no previous
        assert w is None
        assert len(block.inputs) == 0
        block.inputs.append((None, torch.ones((2, 1), device = local_rank)))

        assert len(block.activations) == 0
        block.forward()
        assert len(block.activations) == 1
        assert len(block.act_to_send) == 1
        assert len(block.inputs) == 0
        assert len(block.inputs_to_keep) == 1

        assert torch.allclose(block.activations[0], torch.full((2, 2), 2.0, device = local_rank))

        w = block.send_forward() # Should be matched by a recv_forward
        assert w is None
        assert len(block.act_to_send) == 0
        
        w = block.recv_backward(1)
        assert w is None
        assert len(block.grads) == 1

        y = block.backward()
        assert y is None # only the last block should return its result
        assert len(block.grads) == 0
        assert len(block.activations) == 0
        assert len(block.inputs_to_keep) == 0
        assert len(block.grads_to_send) == 1

        w = block.send_backward()
        assert w is None
        
    elif rank == 1:
        assert block.previous == 0
        assert block.next is None

        block.model.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], device = local_rank))

        assert len(block.inputs) == 0

        w = block.recv_forward(1)
        assert w is None
        assert len(block.inputs) == 1

        assert len(block.activations) == 0
        y = block.forward()
        assert len(block.activations) == 1
        assert torch.allclose(y, torch.full((2, 3), 6.0, device = local_rank))

        assert len(block.act_to_send) == 0 # no next block, nothing to send
        w = block.send_forward() # does nothing
        assert w is None
        assert len(block.activations) == 1
        assert len(block.act_to_send) == 0

        w = block.recv_backward(1) # does nothing either
        assert w is None
        assert len(block.grads) == 0
        assert len(block.activations) == 1

        block.grads.append((None, torch.ones((2, 3), device = local_rank)))

        block.backward()
        assert len(block.grads) == 0
        assert len(block.activations) == 0
        assert len(block.inputs_to_keep) == 0
        assert len(block.grads_to_send) == 1

        w = block.send_backward()
        assert w is None
        assert len(block.grads_to_send) == 0

@pytest.mark.multi
def test_pipeline_creation_multi(init_dist):
    rank, local_rank, world_size = init_dist

    if world_size < 4:
        pytest.skip(f'This test needs 4 gpus (and 4 processes) to run.')
        return

    # Test automatic placement + partitioning
    model = nn.Sequential(
        nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1)
    )
    pipe = Pipeline(model)

    # Test predefined placement
    pipe = Pipeline(model, placement = [2, 3])
    if rank == 2 or rank == 3: assert len(pipe.blocks) == 1
    else: assert len(pipe.blocks) == 0

    pipe = Pipeline(model, placement = [1, 2, 1, 2])
    if rank == 1 or rank == 2: assert len(pipe.blocks) == 2
    else: assert len(pipe.blocks) == 0

    pipe = Pipeline(model, placement = [1, 2, 2, 3])
    if rank == 0: assert len(pipe.blocks) == 0
    else: assert len(pipe.blocks) == 1 # Blocks should be merged together

    # Test predefined partition
    layers = [l for l in model.children()] 
    pipe = Pipeline(layers, partition = None)
    assert len(pipe.blocks) == 1 # they get merged
    assert pipe.placement == list(range(len(model))) # default value

    # Default
    pipe = Pipeline(model)
    assert len(pipe.blocks) == 1 # On every device, there is 1 block
    assert pipe.blocks[0].id == pipe.blocks[0].rank == rank

    # Handmade placement
    pipe = Pipeline(model, placement = [0, 1, 0, 1])
    if rank == 0:
        assert len(pipe.blocks) == 2
        b1, b2 = pipe.blocks[0], pipe.blocks[1]
        assert b1.rank == b2.rank == rank
        assert b1.id == 0
        assert b1.previous is None
        assert b1.next == 1

        assert b2.id == 2
        assert b2.previous == 1
        assert b2.next == 1
    elif rank == 1:
        assert len(pipe.blocks) == 2
        b1, b2 = pipe.blocks[0], pipe.blocks[1]
        assert b1.rank == b2.rank == rank
        assert b1.id == 1
        assert b1.previous == 0
        assert b1.next == 0

        assert b2.id == 3
        assert b2.previous == 0
        assert b2.next is None
    else:
        assert len(pipe.blocks) == 0

    # Handmade partition
    model = nn.Linear(1, 1)
    pipe = Pipeline([model], partition = None)
    assert len(pipe.blocks) == 1
    assert pipe.blocks[0].id == pipe.blocks[0].rank == rank

@pytest.mark.multi
def test_pipe_correctness_multi(init_dist):
    rank, local_rank, world_size = init_dist

    if world_size < 4:
        pytest.skip(f'This test needs at least 4 gpus (and processes) to run.')
        return

    model = nn.Linear(3, 3, bias = False).cuda()
    model.weight.data = torch.full((3, 3), rank + 1, device = local_rank, dtype = torch.float32)
    pipe = Pipeline([model], placement = [0, 1, 2, 3], partition = None)
    last = 3

    inputs = torch.randn((4, 3), device = local_rank)
    targets = torch.randn((4, 3), device = local_rank)
    y, loss = pipe(inputs, targets, loss_fn = nn.functional.mse_loss)
    if rank == last:
        dist.send(y, 0)
        dist.send(loss, 0)
        dist.send(targets, 0)
    elif rank == 0:
        y = torch.empty_like(targets)
        loss = torch.empty((1), device = local_rank)
        dist.recv(y, last)
        dist.recv(loss, last)
        dist.recv(targets, last)

    # Everything should be cleared after a full pass
    for b in pipe.blocks:
        assert len(b.inputs) == 0
        assert len(b.inputs_to_keep) == 0
        assert len(b.activations) == 0
        assert len(b.act_to_send) == 0
        assert len(b.grads) == 0
    
    all_weights = [torch.empty_like(model.weight) for _ in range(world_size)] if rank == 0 else None
    all_grads = [torch.empty_like(model.weight) for _ in range(world_size)] if rank == 0 else None
    dist.gather(model.weight, all_weights, dst = 0)
    dist.gather(model.weight.grad, all_grads, dst = 0)
    if rank == 0:
        model_full = nn.Sequential()
        for w in all_weights:
            l = nn.Linear(3, 3, bias = False)
            l.weight.data = w.data
            model_full.append(l)
            
        y_true = model_full(inputs)
        assert torch.allclose(y, y_true)

        loss_true = nn.functional.mse_loss(y_true, targets, reduction = "sum")
        assert torch.allclose(loss_true, loss)

        loss_true.backward()

        for g, l in zip(all_grads, model_full.children()):
            assert torch.allclose(g.data, l.weight.grad.data)
