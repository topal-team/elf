#!/bin/bash

torchrun --nproc-per-node 4 tests/distributed/distributed_test_block.py
torchrun --nproc-per-node 4 tests/distributed/distributed_test_pipeline.py
