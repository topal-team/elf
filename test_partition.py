import sys
import time
import torch
import numpy as np
from pipeline.partitioners import partition_graph
from models.simple import SimpleTransformer
from settings import model, inputs

if __name__ == '__main__':
    sample = inputs
    # model = SimpleTransformer(500, 256, 12)
    # sample = model.get_sample(2)
    model.eval() # disable randomness, e.g. dropout

    mode = sys.argv[1] if len(sys.argv) > 1 else "default"
    
    trace = torch.fx.symbolic_trace(model)

    parts, inputs, outputs = partition_graph(trace, 4, sample, mode = mode)
    
    for i,p in enumerate(parts):
        print(f'Part {i} needs inputs {inputs[i]} and has outputs {outputs[i]}.')
    
    times = []
    sizes = []
    x = sample
    x = {i: sample for i in inputs[0]}
    for i,s in enumerate(parts):
        start = time.time()
        x = s(**x)
        times.append(time.time() - start)
        # x = {y:z for y,z in x.items() if i + 1 == len(inputs) or y in inputs[i + 1]}
        sizes.append(sum(t.numel() * t.element_size() / (2**20) for t in x.values()))

    print("Execution times:", ["%.3fs" % t for t in times])
    print(f'Median : {np.median(times):.3f} - Stddev : {np.std(times):.3f}')
    print("Transfer sizes:", ["%.2fMB" % s for s in sizes])
    print(f'Median : {np.median(sizes[:-1]):.3f} - Stddev : {np.std(sizes[:-1]):.3f}')

    gt = model(sample)
    for y in x.values():
        assert torch.allclose(y, gt)
