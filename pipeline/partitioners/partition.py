import numpy as np
import torch
from pipeline.utils import Timer
from .custom import split_graph, split_graph_constrained
from .metis import split_graph_metis
from .dagP import split_graph_dagP

import logging
logger = logging.getLogger("partition")

# Some edges should not be part of the cut; for instance values that are not torch.Tensor are a problem to send/recv for pipelining.
# To avoid that we put a soft constraint by giving them a huge communication cost.
NON_TENSOR = 2 << 29

def add_profiling_to_graph(graph_module, iters = 10):
    '''
    Torch FX graph manipulation to add profiling of time & memory to each node.
    This function returns a new module that will profile the operations it runs.
    It also returns a dictionary for time and one for memories, where the profiling results will be stored with the format {node_name: value}
    '''
    def wrap_function(node):
        original_target = node.target
        def timed_forward(*args, **kwargs):
            times = []
            for _ in range(iters):
                with Timer() as timer:
                    output = original_target(*args, **kwargs)
                times.append(timer.time())
            node_time[node.target.__name__] = np.median(times)
            if isinstance(output, torch.Tensor):
                node_memory[node.target.__name__] = output.numel() * output.element_size()
            setattr(node, "target", original_target) # restore original op
            return output
        setattr(node, "target", timed_forward)

    def wrap_module(module, node_name):
        original_forward = module.forward
        def timed_forward(*args, **kwargs):
            times = []
            for _ in range(iters):
                with Timer() as timer:
                    output = original_forward(*args, **kwargs)
                times.append(timer.time())
            node_time[node_name] = np.median(times)
            if isinstance(output, torch.Tensor):
                node_memory[node_name] = output.numel() * output.element_size()
            setattr(module, "forward", original_forward) # restore
            return output
        module.forward = timed_forward
    
    def wrap_method(instance, method_name):
        def timed_method(*args, **kwargs):
            method = getattr(args[0], method_name)
            times = []
            for _ in range(iters):
                with Timer() as timer:
                    output = method(*args, **kwargs)
                times.append(timer.time())
            node_time[instance.name] = np.median(times)
            if isinstance(output, torch.Tensor):
                node_memory[instance.name] = output.numel() * output.element_size()
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

def profile_operations(graph_module, input_sample):
    '''
    Get time and memory for each node of a traced module, when running forward on a sample.
    Returns 2 dicts, one for time and one for memory used respectively. Each one has the format {node_name: value}
    '''
    # Warmup
    with torch.no_grad():
        graph_module(input_sample)

    # Add timing to the graph
    graph_module, node_time, node_memory = add_profiling_to_graph(graph_module)

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

def create_subgraph(graph_module, nodes, inputs, outputs):
    '''
    Creates a module from one block of a partition.
    Returns a GraphModule that can be used like a nn.Module
    The module takes as input a dictionary {inputs1: value1, inputs2: value2, ..} where the keys are the values in parameter inputs.
    It returns as output a similar dictionary, with keys from parameter outputs.
    '''
    subgraph = torch.fx.Graph()
    env = {}

    def load_arg(a):
        return env[a.name]

    with subgraph.inserting_before():
        for i in inputs:
            node = subgraph.placeholder(i)
            env[node.name] = node

    for node in nodes:
        env[node.name] = subgraph.node_copy(node, load_arg)

    with subgraph.inserting_after():
        subgraph.output({
            o: env[o] for o in outputs
        })
        
    return torch.fx.GraphModule(graph_module, subgraph)

def get_inputs_outputs(parts):
    '''
    Finds the dependencies between each block of a partition.
    Returns 2 dicts, one for inputs and one for outputs, respectively. Each one has the format: {partition_idx: [target1, target2, ..]} where targets are node names.
    '''
    inputs = {i: set() for i in range(len(parts))}
    outputs = {i: set() for i in range(len(parts))}

    def find_idx(node):
        for i, p in enumerate(parts):
            if node in p:
                return i

    for i, part in enumerate(parts):
        for node in part:
            if node.op == "placeholder":
                inputs[i].add(node.name)
                part.remove(node)
                continue
            elif node.op == "output":
                outputs[i].add(node.args[0].name)
                part.remove(node)
                continue
            for dep in node.all_input_nodes:
                if dep not in part:
                    inputs[i].add(dep.name)
                    outputs[find_idx(dep)].add(dep.name)
    
    return inputs, outputs


def partition_graph(model, n, sample, mode = "default"):
    '''
    Splits a graph into n parts of roughly equal time.
    Different modes are available: 
    - default: does not take into account memory, no constraint on the number of inputs/outputs
    - constrained: does not take into account memory, inputs & outputs of each block are limited to 1 tensor
    - metis: uses METIS to minimize both time and communication memory. No hard constraint on inputs/outputs.
    - dagP: like METIS, but uses dagP to enforce acyclicity of partition.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trace = torch.fx.symbolic_trace(model.to(device))
    sample = sample.to(device)
    times, memories = profile_operations(trace, sample)

    if mode == "default":
        parts = split_graph(trace, times, memories, n)
    elif mode == "constrained":
        parts = split_graph_constrained(trace, times, memories, n)
    elif mode == "metis":
        parts = split_graph_metis(trace, times, memories, n)        
    elif mode == "dagP":
        parts = split_graph_dagP(trace, times, memories, n)
    else:
        raise Exception("Unknown graph partitioning mode : {mode}.\n\
                        Available modes:\n\t\
                        - default: does not take into account memory, no constraint on the number of inputs/outputs\n\t\
                        - constrained: does not take into account memory, inputs & outputs of each block are limited to 1 tensor\n\t\
                        - metis: uses METIS to minimize both time and communication memory. No hard constraint on inputs/outputs.\n\t\
                        - dagP: like METIS, but uses dagP to enforce acyclicity of partition.")
    
    inputs, outputs = get_inputs_outputs(parts)
    blocks = []
    for i, p in enumerate(parts):
        graph = create_subgraph(trace, p, inputs[i], outputs[i])
        blocks.append(graph)

    return blocks, inputs, outputs