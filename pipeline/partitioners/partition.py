import torch
from .profile import profile_operations
from .custom import split_graph, split_graph_constrained
from .metis import split_graph_metis
from .dagP import split_graph_dagP
from ..utils import TimerCPU

import logging
logger = logging.getLogger("partition")


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

def get_inputs_outputs_single(part):
    '''
    Get the input/output nodes of a single part from the graph
    Useful for already splitted graphs
    '''
    inputs = set()
    outputs = set()
    nodes = list(part.graph.node)
    for node in nodes:
        if node.op == "placeholder":
            inputs.append(node.name)
        if node.op == "output":
            outputs.append(node.name)
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

    trace = torch.fx.symbolic_trace(model)
    
    with TimerCPU() as timer:
        times, memories = profile_operations(trace, sample, device)
    logger.info(f'Time taken to profile graph : {timer.time():.3f}s')

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

    logger.info(f'Estimated times : {[sum([times.get(n.name, 0) for n in part]) for part in parts]}')
    logger.info(f'Memory transfers : {[sum([memories.get(o, 0) for o in out]) for out in outputs.values()]}')
    return blocks, inputs, outputs
