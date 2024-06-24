import gc
import torch
import numpy as np
import torch.nn as nn
from pipeline.utils import Timer

import logging
logger = logging.getLogger('profiler')

# Some edges should not be part of the cut; for instance values that are not torch.Tensor are a problem to send/recv for pipelining.
# To avoid that we put a soft constraint by giving them a huge communication cost.
NON_TENSOR = 2 << 29

def add_profiling_to_graph(graph_module, device, iters = 10):
    '''
    Torch FX graph manipulation to add profiling of time & memory to each node.
    This function returns a new module that will profile the operations it runs.
    It also returns a dictionary for time and one for memories, where the profiling results will be stored with the format {node_name: value}
    '''
    def wrap_function(node):
        original_target = node.target
        def timed_forward(*args, **kwargs):
            times = []
            args = [t.to(device) if isinstance(t, torch.Tensor) else t for t in args]
            kwargs = {x:y.to(device) if isinstance(y, torch.Tensor) else y for x,y in kwargs.items()}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            for _ in range(iters):
                with Timer() as timer:
                    output = original_target(*args, **kwargs)
                times.append(timer.time())
                            
            node_time[node.target.__name__] = np.median(times)
            if isinstance(output, torch.Tensor):
                node_memory[node.target.__name__] = output.numel() * output.element_size()
                output = output.to('cpu', non_blocking = True)
            
            setattr(node, "target", original_target) # restore original op
            return output
        
        setattr(node, "target", timed_forward)

    def wrap_module(module, node_name):
        original_forward = module.forward
        def timed_forward(*args, **kwargs):
            nonlocal module
            times = []
            module = module.to(device)
            args = [t.to(device) if isinstance(t, torch.Tensor) else t for t in args]
            kwargs = {x:y.to(device) if isinstance(y, torch.Tensor) else y for x,y in kwargs.items()}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            for _ in range(iters):
                with Timer() as timer:
                    output = original_forward(*args, **kwargs)
                times.append(timer.time())
                
            
            node_time[node_name] = np.median(times)
            if isinstance(output, torch.Tensor):
                node_memory[node_name] = output.numel() * output.element_size()
                output = output.to('cpu', non_blocking = True)
                
            module.cpu()
                
            setattr(module, "forward", original_forward) # restore
            return output
        
        module.forward = timed_forward
    
    def wrap_method(instance, method_name):
        def timed_method(*args, **kwargs):
            method = getattr(args[0], method_name)
            times = []
            args = [t.to(device) if isinstance(t, torch.Tensor) else t for t in args]
            kwargs = {x:y.to(device) if isinstance(y, torch.Tensor) else y for x,y in kwargs.items()}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            for _ in range(iters):
                with Timer() as timer:
                    output = method(*args, **kwargs)
                times.append(timer.time())
            
            node_time[instance.name] = np.median(times)
            if isinstance(output, torch.Tensor):
                node_memory[instance.name] = output.numel() * output.element_size()
                output = output.to('cpu', non_blocking = True)
            
            setattr(instance, method_name, method) # restore
            return output
        
        setattr(instance, method_name, timed_method)

    node_time = {}
    node_memory = {}

    for node in graph_module.graph.nodes:
        if node.op == 'call_function':
            wrap_function(node)

        elif node.op == 'call_module':
            submodule = dict(graph_module.named_modules())[node.target]
            wrap_module(submodule, node.name)

        elif node.op == 'call_method':
            instance = node.args[0]
            wrap_method(instance, node.target)

        elif node.op == 'get_attr':
            ... # Is it really useful to profile these ones ?

    graph_module.recompile()
    return graph_module, node_time, node_memory

def profile_operations(model, input_sample, device = 'cuda'):
    '''
    Get time and memory for each node of a traced module, when running forward on a sample.
    Returns 2 dicts, one for time and one for memory used respectively. Each one has the format {node_name: value}
    '''
    # Add timing to the graph
    graph_module, node_time, node_memory = add_profiling_to_graph(model, device)

    # Execute the modified graph
    with torch.no_grad():
        graph_module(input_sample)

    # Fill the operations that were not timed
    for node in graph_module.graph.nodes:
        if node.name not in node_time:
            node_time[node.name] = 0
        if node.name not in node_memory:
            node_memory[node.name] = NON_TENSOR # if it's not a tensor, don't cut here as it creates problems for comms

    return node_time, node_memory
