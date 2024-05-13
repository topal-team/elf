import os
from ..schedule import *
import pytest

@pytest.mark.single
def test_afab():
    placement = [0, 1]
    n_micro_batches = 2

    schedule = generate_afab_schedule(placement, n_micro_batches)
    print(schedule)

    assert schedule == [
        Operation(0, 0, Operations.RECV_FORWARD, 0), Operation(0, 0, Operations.FORWARD, 0), Operation(0, 0, Operations.SEND_FORWARD, 0),
        Operation(0, 1, Operations.RECV_FORWARD, 0), Operation(0, 1, Operations.FORWARD, 0), Operation(0, 1, Operations.SEND_FORWARD, 0),
        Operation(0, 0, Operations.RECV_BACKWARD, 0), Operation(0, 0, Operations.BACKWARD, 0), Operation(0, 0, Operations.SEND_BACKWARD, 0),
        Operation(0, 1, Operations.RECV_BACKWARD, 0), Operation(0, 1, Operations.BACKWARD, 0), Operation(0, 1, Operations.SEND_BACKWARD, 0),

        Operation(1, 0, Operations.RECV_FORWARD, 1), Operation(1, 0, Operations.FORWARD, 1), Operation(1, 0, Operations.SEND_FORWARD, 1),
        Operation(1, 1, Operations.RECV_FORWARD, 1), Operation(1, 1, Operations.FORWARD, 1), Operation(1, 1, Operations.SEND_FORWARD, 1),
        Operation(1, 0, Operations.RECV_BACKWARD, 1), Operation(1, 0, Operations.BACKWARD, 1), Operation(1, 0, Operations.SEND_BACKWARD, 1),
        Operation(1, 1, Operations.RECV_BACKWARD, 1), Operation(1, 1, Operations.BACKWARD, 1), Operation(1, 1, Operations.SEND_BACKWARD, 1),
    ]

@pytest.mark.single
@pytest.mark.skip(reason = "Not modified yet")
def test_1f1b():
    placement = [0, 1]
    n_micro_batches = 4

    schedule = generate_1f1b_schedule(placement, n_micro_batches)
    schedule0 = list(filter(lambda op: op.rank == 0, schedule))

    assert schedule0 == [
        Operation(0, 0, Operations.RECV_FORWARD, 0), Operation(0, 0, Operations.FORWARD, 0), Operation(0, 0, Operations.SEND_FORWARD, 0),
        Operation(0, 0, Operations.RECV_FORWARD, 1), Operation(0, 0, Operations.FORWARD, 1), Operation(0, 0, Operations.SEND_FORWARD, 1),
        Operation(0, 0, Operations.RECV_BACKWARD, 0), Operation(0, 0, Operations.BACKWARD, 0), Operation(0, 0, Operations.SEND_BACKWARD, 0),
        Operation(0, 0, Operations.RECV_FORWARD, 2), Operation(0, 0, Operations.FORWARD, 2), Operation(0, 0, Operations.SEND_FORWARD, 2),
        Operation(0, 0, Operations.RECV_BACKWARD, 1), Operation(0, 0, Operations.BACKWARD, 1), Operation(0, 0, Operations.SEND_BACKWARD, 1),
        Operation(0, 0, Operations.RECV_FORWARD, 3), Operation(0, 0, Operations.FORWARD, 3), Operation(0, 0, Operations.SEND_FORWARD, 3),
        Operation(0, 0, Operations.RECV_BACKWARD, 2), Operation(0, 0, Operations.BACKWARD, 2), Operation(0, 0, Operations.SEND_BACKWARD, 2),
        Operation(0, 0, Operations.RECV_BACKWARD, 3), Operation(0, 0, Operations.BACKWARD, 3), Operation(0, 0, Operations.SEND_BACKWARD, 3)
    ]

    schedule1 = list(filter(lambda op: op.rank == 1, schedule))

    # Send/Recv are swapped on odd devices to avoid deadlocks :)
    assert schedule1 == [
        Operation(1, 1, Operations.RECV_FORWARD, 0), Operation(1, 1, Operations.FORWARD, 0), Operation(1, 1, Operations.RECV_BACKWARD, 0),
        Operation(1, 1, Operations.SEND_FORWARD), Operation(1, 1, Operations.BACKWARD), Operation(1, 1, Operations.RECV_FORWARD),
        Operation(1, 1, Operations.SEND_BACKWARD), Operation(1, 1, Operations.FORWARD), Operation(1, 1, Operations.RECV_BACKWARD),
        Operation(1, 1, Operations.SEND_FORWARD), Operation(1, 1, Operations.BACKWARD), Operation(1, 1, Operations.RECV_FORWARD),
        Operation(1, 1, Operations.SEND_BACKWARD), Operation(1, 1, Operations.FORWARD), Operation(1, 1, Operations.RECV_BACKWARD),
        Operation(1, 1, Operations.SEND_FORWARD), Operation(1, 1, Operations.BACKWARD), Operation(1, 1, Operations.RECV_FORWARD),
        Operation(1, 1, Operations.SEND_BACKWARD), Operation(1, 1, Operations.FORWARD), Operation(1, 1, Operations.RECV_BACKWARD),
        Operation(1, 1, Operations.SEND_FORWARD), Operation(1, 1, Operations.BACKWARD), Operation(1, 1, Operations.SEND_BACKWARD),
    ]
