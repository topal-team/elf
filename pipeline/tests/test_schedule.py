import os
import torch
from ..schedule import *

def test_afab():
    placement = [0, 1]
    n_micro_batches = 2

    os.environ["RANK"] = "0"
    schedule = generate_afab_schedule(placement, n_micro_batches)

    assert schedule == [
        (0, Operations.RECV_FORWARD), (0, Operations.FORWARD), (0, Operations.SEND_FORWARD),
        (0, Operations.RECV_FORWARD), (0, Operations.FORWARD), (0, Operations.SEND_FORWARD),
        (0, Operations.RECV_BACKWARD), (0, Operations.BACKWARD), (0, Operations.SEND_BACKWARD),
        (0, Operations.RECV_BACKWARD), (0, Operations.BACKWARD), (0, Operations.SEND_BACKWARD),
    ]

    os.environ["RANK"] = "1"
    schedule = generate_afab_schedule(placement, n_micro_batches)

    assert schedule == [
        (1, Operations.RECV_FORWARD), (1, Operations.FORWARD), (1, Operations.SEND_FORWARD),
        (1, Operations.RECV_FORWARD), (1, Operations.FORWARD), (1, Operations.SEND_FORWARD),
        (1, Operations.RECV_BACKWARD), (1, Operations.BACKWARD), (1, Operations.SEND_BACKWARD),
        (1, Operations.RECV_BACKWARD), (1, Operations.BACKWARD), (1, Operations.SEND_BACKWARD),
    ]

def test_1f1b():
    placement = [0, 1]
    n_micro_batches = 4

    os.environ["RANK"] = "0"
    schedule = generate_1f1b_schedule(placement, n_micro_batches)

    assert schedule == [
        (0, Operations.RECV_FORWARD), (0, Operations.FORWARD), (0, Operations.SEND_FORWARD),
        (0, Operations.RECV_FORWARD), (0, Operations.FORWARD), (0, Operations.SEND_FORWARD),
        (0, Operations.RECV_BACKWARD), (0, Operations.BACKWARD), (0, Operations.SEND_BACKWARD),
        (0, Operations.RECV_FORWARD), (0, Operations.FORWARD), (0, Operations.SEND_FORWARD),
        (0, Operations.RECV_BACKWARD), (0, Operations.BACKWARD), (0, Operations.SEND_BACKWARD),
        (0, Operations.RECV_FORWARD), (0, Operations.FORWARD), (0, Operations.SEND_FORWARD),
        (0, Operations.RECV_BACKWARD), (0, Operations.BACKWARD), (0, Operations.SEND_BACKWARD),
        (0, Operations.RECV_BACKWARD), (0, Operations.BACKWARD), (0, Operations.SEND_BACKWARD)
    ]

    os.environ["RANK"] = "1"
    schedule = generate_1f1b_schedule(placement, n_micro_batches)

    # Send/Recv are swapped on odd devices to avoid deadlocks :)
    assert schedule == [
        (1, Operations.RECV_FORWARD), (1, Operations.FORWARD), (1, Operations.RECV_BACKWARD),
        (1, Operations.SEND_FORWARD), (1, Operations.BACKWARD), (1, Operations.RECV_FORWARD),
        (1, Operations.SEND_BACKWARD), (1, Operations.FORWARD), (1, Operations.RECV_BACKWARD),
        (1, Operations.SEND_FORWARD), (1, Operations.BACKWARD), (1, Operations.RECV_FORWARD),
        (1, Operations.SEND_BACKWARD), (1, Operations.FORWARD), (1, Operations.RECV_BACKWARD),
        (1, Operations.SEND_FORWARD), (1, Operations.BACKWARD), (1, Operations.RECV_FORWARD),
        (1, Operations.SEND_BACKWARD), (1, Operations.FORWARD), (1, Operations.RECV_BACKWARD),
        (1, Operations.SEND_FORWARD), (1, Operations.BACKWARD), (1, Operations.SEND_BACKWARD),
    ]