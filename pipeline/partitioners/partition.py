'''
API and main utils for graph partition
'''

import torch
from .profile import profile_operations
from .custom import split_graph, split_graph_constrained
from .metis import split_graph_metis
from .dagP import split_graph_dagP

import logging
logger = logging.getLogger("partition")

def create_subgraph(graph_module, nodes, inputs, outputs):
    '''
    Creates a module from one block of a partition.
    The module takes as input a dictionary {inputs1: value1, inputs2: value2, ..} where the keys are the values in parameter inputs.
    and returns a similar dictionary, with keys from parameter outputs.

    :return: a graph that can be used like a nn.Module
    :rtype: fx.GraphModule
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

    :param parts: partition of a model
    :type parts: List[List[fx.Node]]
    :return: 2 dicts, one for inputs and one for outputs, respectively. Each one has the format: {partition_idx: [target1, target2, ..]} where targets are node names.
    :rtype: Dict[int, List[str]], Dict[int, List[str]]
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

    :param part: one part of a partitioned model
    :type part: List[fx.Node]

    :return: names of the input and output variables
    :rtype: List[str], List[str]
    '''
    inputs = set()
    outputs = set()
    nodes = list(part.graph.nodes)
    for node in nodes:
        if node.op == "placeholder":
            inputs.add(node.name)
        if node.op == "output":
            for name in node.args[0]:
                outputs.add(name)
    return inputs, outputs

def partition_graph(model, n, sample, mode = "default"):
    '''
    Splits a graph into n parts of roughly equal time.

    :param model: torch model
    :type model: nn.Module
    :param n: number of parts to create
    :type n: int
    :param sample: example of inputs to feed to the model (used for profiling)
    :type sample: Tensor
    :param mode: Different modes are available:

        - default: does not take into account memory, no constraint on the number of inputs/outputs
        - constrained: does not take into account memory, inputs & outputs of each block are limited to 1 tensor
        - metis: uses METIS to minimize both time and communication memory. No hard constraint on inputs/outputs.
        - dagP: like METIS, but uses dagP to enforce acyclicity of partition.
    
    :type mode: str

    :return: 

        - ``n`` new modules corresponding to the partition
        - name of input variables for each module. Each one of them takes its inputs as named parameters with these names
        - name of output variables for each module. Each one of them outputs a dictionary with these names as keys

    :rtype: List[fx.GraphModule], List[List[str]], List[List[str]]
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

    estimated_times = [sum([times.get(n.name, 0) for n in part]) for part in parts]
    estimated_mems = [sum([memories.get(o, 0) for o in out]) / (2**20) for out in outputs.values()]
    logger.info(f'Estimated times : {["%.3fs" % t for t in estimated_times]}')
    logger.info(f'Estimated memory transfers : {["%.1fMB" % t for t in estimated_mems]}')
    return blocks, inputs, outputs
