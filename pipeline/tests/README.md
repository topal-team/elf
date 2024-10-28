## How to run tests

### All tests
Use script `scripts/test.sh` to run all tests, or run:
```bash
pytest -m single .
torchrun --nproc-per-node 4 -m pytest -m multi .
```

### Run a specific test
```bash
pytest -k test-name
```

## Graph extraction
Output of extracted graph is the same as initial module
``model(x) == graph.module()(x)``
and for weights grads too. 

## Profiling
``len(times) == len(fx_nodes)``  \
for all placeholders/outputs, ``time = 0``  \
after profiling, all parameters and buffers are the same  \
check that outputs/grads are the same after profiling (graph not modified) 

## Graph processing
check that output/grads are the same as before inplace removal 

## Partition
Number of parts == number of parts asked for  \
Check that output  of sequentially computing the parts is the same as the original model \
Every part should have 1 input and 1 output node  \
Number of computational nodes is the same for original model and sum of parts.  \
Output names of module $i$ = Input names of module $i+1$ 

## Schedule
$\#fwd=\#bwd=\#mb\times pp$  \
(and same for recv and send)  \
$\forall mb\_id$, $\#fwd = \#bwd = pp$  \
$\forall mb\_id$, $recv\_fwd < fw < send\_fwd < recv\_bwd < bwd < send\_bwd$ 
$\forall bwd < all\_reduce$  \
$\#all\_reduce = pp$ 

## Pipeline
Output/Gradients of forward == normal model , for different mb_sizes \
Parameters are correctly updated with dp  \
No memory leakage