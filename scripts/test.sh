#!/bin/bash -l

pytest -m single . $*
torchrun --nproc-per-node 4 -m pytest -m multi . $*
