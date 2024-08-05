import sys
import time
import torch
import numpy as np
sys.path.append('./')
from pipeline.partitioners import *
from pipeline.utils import Timer, TimerCPU
from models.simple import SimpleTransformer, SimpleCNN
from argparse import ArgumentParser

import logging
logger = logging.getLogger(f'test partition')
logging.basicConfig(level = logging.INFO)

def pretty_print_params(n):
    if n > 1e9:
        return f'{n/1e9:.1f}B'
    elif n > 1e6:
        return f'{n/1e6:.1f}M'
    else:
        return f'{int(n)}'

def compare_partitioners(model, sample):
    n = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trace = torch.fx.symbolic_trace(model)
    with TimerCPU() as timer:
        times, memories = profile_operations(trace, sample)
    print(f'Time taken to profile  : {timer.time():.3f}s')
    n_profiled_mem = len(memories.keys())
    n_profiled_mem_nz = len([v for v in memories.values() if v != 0])
    n_profiled_t = len(times.keys())
    n_profiled_t_nz = len([v for v in times.values() if v != 0])
    print(f'Proportion of nodes profiled :')
    print(f'\tMem - {(n_profiled_mem / len(trace.graph.nodes)) * 100:.2f}% (Non-Zero : {(n_profiled_mem_nz / len(trace.graph.nodes)) * 100:.2f}%)')
    print(f'\tTime - {(n_profiled_t / len(trace.graph.nodes)) * 100:.2f}% (Non-Zero: {(n_profiled_t_nz / len(trace.graph.nodes)) * 100:.2f}%)')

    for name, partitioner in zip(['default', 'constrained', 'metis', 'dagP'],
                                 [split_graph, split_graph_constrained, split_graph_metis, split_graph_dagP]):
        print(f'-- {name} --')
        with TimerCPU() as timer:
            parts = partitioner(trace, times, memories, n)
        print(f'\tTime taken to partition  : {timer.time():.3f}s')
        
        while len(parts) != n: parts.append([])

        inputs, outputs = get_inputs_outputs(parts)

        estimated_times = [sum(np.median(times[node.name]) for node in p) for p in parts]
        memories_used = [sum([memories[o] for o in out]) / (2<<20) for out in outputs.values()]

        blocks = []
        for i, p in enumerate(parts):
            graph = create_subgraph(trace, p, inputs[i], outputs[i])
            blocks.append(graph)

        real_times = []
        sample = sample.to(device)
        x = {i: sample for i in inputs[0]}
        for i,s in enumerate(blocks):
            s = s.to(device)
            with Timer() as timer:
                with torch.no_grad():
                    x = s(**x)
            real_times.append(timer.time())
            x = {y:z for y,z in x.items() if i + 1 == len(inputs) or y in inputs[i + 1]}
            s = s.to('cpu', non_blocking = True)

        print(f'\tEstimated times : {["%.3fs" % t for t in estimated_times]}')
        print(f'\tTime measured : {["%.3fs" % t for t in real_times]}')
        print(f'\tMedian : {np.median(real_times):.3f}s - Stddev : {np.std(real_times):.3f}')
        print(f'\tMemory sent : {["%.1fMB" % m for m in memories_used]} (total = {sum(memories_used[:-1]):.1f}MB)\n')

if __name__ == '__main__':
    parser = ArgumentParser(description = "Demo/Test of pipelined model")
    parser.add_argument('--mode', choices=['default', 'constrained', 'metis', 'dagP'], default = 'default', required = False, help='partition mode')
    parser.add_argument('--log', choices=['debug', 'info', 'none'], default='info', required=False, help="logging level")
    parser.add_argument('--model', '-m', choices=["gpt", "tf", "cnn"], required=False, default="tf")
    args = parser.parse_args()
    match args.log:
        case 'debug':
            logging.getLogger().setLevel(logging.DEBUG)
        case 'info':
            logging.getLogger().setLevel(logging.INFO)
        case 'none':
            logging.getLogger().setLevel(100)

    match args.model:
        case 'gpt':
            from settings import model, inputs
            sample = inputs
        case 'tf':
            model = SimpleTransformer(200, 128, 4)
            sample = model.get_sample(16)
        case 'cnn':
            model = SimpleCNN()
            sample = model.get_sample(32)

    print('# of trainable parameters : ', pretty_print_params(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    compare_partitioners(model, sample)

