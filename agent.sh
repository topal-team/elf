#!/bin/bash

wandb offline
export WANDB_MODE=offline
torchrun --nproc-per-node 4 --nnodes 1 --standalone --no-python wandb agent --count 5 felisamici/metis-estimate/8wn02wbt
