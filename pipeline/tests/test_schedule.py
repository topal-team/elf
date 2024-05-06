import os
from ..schedule import *
import pytest

@pytest.mark.single
def test_afab():
    placement = [0, 1]
    n_micro_batches = 2

    os.environ["RANK"] = "0"
    schedule = generate_afab_schedule(placement, n_micro_batches)

    assert schedule == [
        (0, Operations.RECV_FORWARD, 0), (0, Operations.FORWARD, 0), (0, Operations.SEND_FORWARD, 0),
        (0, Operations.RECV_FORWARD, 1), (0, Operations.FORWARD, 1), (0, Operations.SEND_FORWARD, 1),
        (0, Operations.RECV_BACKWARD, 0), (0, Operations.BACKWARD, 0), (0, Operations.SEND_BACKWARD, 0),
        (0, Operations.RECV_BACKWARD, 1), (0, Operations.BACKWARD, 1), (0, Operations.SEND_BACKWARD, 1),
    ]

    os.environ["RANK"] = "1"
    schedule = generate_afab_schedule(placement, n_micro_batches)

    assert schedule == [
        (1, Operations.RECV_FORWARD, 0), (1, Operations.FORWARD, 0), (1, Operations.SEND_FORWARD, 0),
        (1, Operations.RECV_FORWARD, 1), (1, Operations.FORWARD, 1), (1, Operations.SEND_FORWARD, 1),
        (1, Operations.RECV_BACKWARD, 0), (1, Operations.BACKWARD, 0), (1, Operations.SEND_BACKWARD, 0),
        (1, Operations.RECV_BACKWARD, 1), (1, Operations.BACKWARD, 1), (1, Operations.SEND_BACKWARD, 1),
    ]

@pytest.mark.single
@pytest.mark.skip(reason = "not modified yet")
def test_1f1b():
    placement = [0, 1]
    n_micro_batches = 4

    os.environ["RANK"] = "0"
    schedule = generate_1f1b_schedule(placement, n_micro_batches)

    assert schedule == [
        (0, Operations.RECV_FORWARD, 0), (0, Operations.FORWARD, 0), (0, Operations.SEND_FORWARD, 0),
        (0, Operations.RECV_FORWARD, 1), (0, Operations.FORWARD, 1), (0, Operations.SEND_FORWARD, 1),
        (0, Operations.RECV_BACKWARD, 0), (0, Operations.BACKWARD, 0), (0, Operations.SEND_BACKWARD, 0),
        (0, Operations.RECV_FORWARD, 2), (0, Operations.FORWARD, 2), (0, Operations.SEND_FORWARD, 2),
        (0, Operations.RECV_BACKWARD, 1), (0, Operations.BACKWARD, 1), (0, Operations.SEND_BACKWARD, 1),
        (0, Operations.RECV_FORWARD, 3), (0, Operations.FORWARD, 3), (0, Operations.SEND_FORWARD, 3),
        (0, Operations.RECV_BACKWARD, 2), (0, Operations.BACKWARD, 2), (0, Operations.SEND_BACKWARD, 2),
        (0, Operations.RECV_BACKWARD, 3), (0, Operations.BACKWARD, 3), (0, Operations.SEND_BACKWARD, 3)
    ]

    os.environ["RANK"] = "1"
    schedule = generate_1f1b_schedule(placement, n_micro_batches)

    # Send/Recv are swapped on odd devices to avoid deadlocks :)
    assert schedule == [
        (1, Operations.RECV_FORWARD, 0), (1, Operations.FORWARD, 0), (1, Operations.RECV_BACKWARD, 0),
        (1, Operations.SEND_FORWARD), (1, Operations.BACKWARD), (1, Operations.RECV_FORWARD),
        (1, Operations.SEND_BACKWARD), (1, Operations.FORWARD), (1, Operations.RECV_BACKWARD),
        (1, Operations.SEND_FORWARD), (1, Operations.BACKWARD), (1, Operations.RECV_FORWARD),
        (1, Operations.SEND_BACKWARD), (1, Operations.FORWARD), (1, Operations.RECV_BACKWARD),
        (1, Operations.SEND_FORWARD), (1, Operations.BACKWARD), (1, Operations.RECV_FORWARD),
        (1, Operations.SEND_BACKWARD), (1, Operations.FORWARD), (1, Operations.RECV_BACKWARD),
        (1, Operations.SEND_FORWARD), (1, Operations.BACKWARD), (1, Operations.SEND_BACKWARD),
    ]
