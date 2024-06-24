import sys
import time
import torch
import numpy as np
from pipeline.partitioners import *
from pipeline.utils import Timer, TimerCPU
from models.simple import SimpleTransformer
from settings import model, inputs
from argparse import ArgumentParser

import logging
logger = logging.getLogger(f'test partition')
logging.basicConfig(level = logging.INFO)

def compare_partitioners(model, sample):
    n = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trace = torch.fx.symbolic_trace(model)
    with TimerCPU() as timer:
        times, memories = profile_operations(trace, sample, device)
    print(f'Time taken to profile  : {timer.time():.3f}s')

    for name, partitioner in zip(['default', 'constrained', 'metis', 'dagP'],
                                 [split_graph, split_graph_constrained, split_graph_metis, split_graph_dagP]):
        print(f'-- {name} --')
        with TimerCPU() as timer:
            parts = partitioner(trace, times, memories, n)
        print(f'\tTime taken to partition  : {timer.time():.3f}s')
        inputs, outputs = get_inputs_outputs(parts)
        memories_used = [sum([memories[o] for o in out]) / (2<<20) for out in outputs.values()]
        blocks = []
        for i, p in enumerate(parts):
            graph = create_subgraph(trace, p, inputs[i], outputs[i])
            blocks.append(graph)

        real_times = []
        sample = sample.cuda()
        x = {i: sample for i in inputs[0]}
        for i,s in enumerate(blocks):
            s = s.to(device)
            torch.cuda.synchronize()
            with Timer() as timer:
                with torch.no_grad():
                    x = s(**x)
            real_times.append(timer.time())
            x = {y:z for y,z in x.items() if i + 1 == len(inputs) or y in inputs[i + 1]}
            s = s.to('cpu', non_blocking = True)

        print(f'\tTime measured : {["%.3fs" % t for t in real_times]}')
        print(f'\tMedian : {np.median(real_times):.3f}s - Stddev : {np.std(real_times):.3f}')
        print(f'\tTotal memory sent : {["%.1fMB" % m for m in memories_used]}\n')

if __name__ == '__main__':
    parser = ArgumentParser(description = "Demo/Test of pipelined model")
    parser.add_argument('--mode', choices=['default', 'constrained', 'metis', 'dagP'], default = 'default', required = False, help='partition mode')
    parser.add_argument('--log', choices=['debug', 'info', 'none'], default='info', required=False, help="logging level")
    parser.add_argument('--compare', '-c', required=False, action='store_true')
    args = parser.parse_args()
    match args.log:
        case 'debug':
            logging.getLogger().setLevel(logging.DEBUG)
        case 'info':
            logging.getLogger().setLevel(logging.INFO)
        case 'none':
            logging.getLogger().setLevel(100)


    sample = inputs
    # model = SimpleTransformer(500, 256, 6)
    # sample = model.get_sample(2)

    if args.compare:
        compare_partitioners(model, sample)
        exit(0)
    
    # trace = torch.fx.symbolic_trace(model)

    start = time.time()
    parts, inputs, outputs = partition_graph(model, 4, sample, mode = args.mode)
    end = time.time()
    print(f'Time taken to partition: {end - start:.3f}s')
    
    for i,p in enumerate(parts):
        print(f'Part {i} needs inputs {inputs[i]} and has outputs {outputs[i]}.')
    
    times = []
    sample = sample.cuda()
    x = {i: sample for i in inputs[0]}
    for i,s in enumerate(parts):
        with Timer() as timer:
            with torch.no_grad():
                x = s(**x)
        times.append(timer.time())
        x = {y:z for y,z in x.items() if i + 1 == len(inputs) or y in inputs[i + 1]}

    print(["%.3f" % t for t in times])
    print(f'Median : {np.median(times):.3f} - Stddev : {np.std(times):.3f}')
