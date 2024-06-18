import os
import time
import tempfile
import numpy as np
import subprocess
import torch
import logging
logger = logging.getLogger("partition")

class Node:
    NON_TENSOR = (2 << 29)
    def __init__(self, node, time, idx):
        self.node = node
        self.time = time
        self.idx = idx
        self.edges = []

    def to_line(self):
        '''
        s w1 w2 ... wncon v1 e1 v2 e2 ... vk ek
        where s is the size of the vertex, w1, w2, . . . , wncon are the ncon vertex weights associated with this vertex, v1, . . . , vk
        are the vertices adjacent to this vertex, and e1, . . . , ek are the weights of these edges.
        http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf - Section 4.1.1
        '''
        line = f'{max(int(self.time * 1e3), 1)}'
        for v,e in self.edges:
            line += f' {v} {max(int(e / 1e3), 1)}'
        return line
    
    def __repr__(self):
        return repr(self.node)

def convert_fx(module, times, memories):
    '''
    Converts a torch FX graph into our custom format using the Node class
    Our format includes informations about the time & memory profiling, and edges between each nodes
    '''
    nodes = list(module.graph.nodes)
    graph = {}
    indices = {}
    for i, node in enumerate(nodes):
        graph[node.name] = Node(node, times[node.name], i + 1)
        indices[node.name] = i + 1
        for dep in node.all_input_nodes:
            graph[dep.name].edges.append((indices[node.name], memories[dep.name])) # add smth to distinguish in and out edges ?
            graph[node.name].edges.append((indices[dep.name], memories[dep.name]))

    return graph

def write_metis(graph):
    '''
    Dumps the info about a custom graph into a file in METIS format
    '''
    file = tempfile.NamedTemporaryFile("w+", dir = ".")
    n = len(graph)
    m = sum(map(lambda n : len(n.edges), list(graph.values()))) // 2 # edges are counted twice
    fmt = '011'
    ncon = 1
    file.write(f'{n} {m} {fmt} {ncon}\n')
    for node in graph.values():
        file.write(node.to_line() + "\n")
    return file

def execute_metis(file, n):
    '''
    Runs METIS on a file
    '''
    file.flush()  # Ensure all data is written before executing
    file.seek(0)  # Reset file pointer to the beginning for reading in subprocess
    subprocess.run(["gpmetis", file.name, str(n), "-objtype=vol", "-contig", "-minconn"], stdout=subprocess.DEVNULL)  # Execute gpmetis with the file and partition into 5 parts

def read_metis(graph, file):
    '''
    Reads the output of Graph Partitioning METIS and breaks a custom graph accordingly.
    '''
    f = open(file, "r")
    
    mapping = []
    nodes = list(graph.values())
    lines = list(map(int, f.readlines()))
    times = []
    parts = []
    n = 0
    
    for l in range(len(lines)):
        i = lines[l]
        # WE NEED TO HAVE CONTIGUOUS PARTITIONS
        # METIS does not enforce that, so we have to fix it by ignoring some attributions
        if i not in mapping and (l < len(lines) - 1 and i == lines[l + 1]):
            mapping.append(i)
            parts.append([])
            times.append(0)
            i = len(mapping) - 1
        else:
            i = len(mapping) - 1
            # i = mapping.index(i)
            
        parts[i].append(nodes[n].node)
        times[i] += nodes[n].time
        assert nodes[n].idx == n + 1
        n += 1

    f.close()
    os.remove(f.name)
    logger.info(f'Estimated times after partition adjustment : {["%.3f" % e for e in times]}')
    return parts

def add_profiling_to_graph(graph_module):
    '''
    Torch FX graph manipulation to add profiling of time & memory to each node.
    This function returns a new module that will profile the operations it runs.
    It also returns a dictionary for time and one for memories, where the profiling results will be stored with the format {node_name: value}
    '''
    def wrap_function(node):
        original_target = node.target
        def timed_forward(*args, **kwargs):
            with Timer() as timer:
                for _ in range(10):
                    output = original_target(*args, **kwargs)
            node_time[node.target.__name__] = timer.time()
            if isinstance(output, torch.Tensor):
                node_memory[node.target.__name__] = output.numel() * output.element_size()
            setattr(node, "target", original_target) # restore original op
            return output
        setattr(node, "target", timed_forward)

    def wrap_module(module, node_name):
        original_forward = module.forward
        def timed_forward(*args, **kwargs):
            with Timer() as timer:

                for _ in range(10):
                    output = original_forward(*args, **kwargs)
            node_time[node_name] = timer.time()
            if isinstance(output, torch.Tensor):
                node_memory[node_name] = output.numel() * output.element_size()
            setattr(module, "forward", original_forward) # restore
            return output
        module.forward = timed_forward
    
    def wrap_method(instance, method_name):
        def timed_method(*args, **kwargs):
            method = getattr(args[0], method_name)
            with Timer() as timer:
                for _ in range(10):
                    output = method(*args, **kwargs)
            assert instance.name not in node_time
            node_time[instance.name] = timer.time()
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
    # Add timing to the graph
    graph_module, node_time, node_memory = add_profiling_to_graph(graph_module)

    # Execute the modified graph
    graph_module(input_sample)

    # Fill the operations that were not timed
    for node in graph_module.graph.nodes:
        if node.name not in node_time:
            node_time[node.name] = 0
        if node.name not in node_memory:
            node_memory[node.name] = Node.NON_TENSOR # if it's not a tensor, don't cut here as it creates problems for comms

    return node_time, node_memory

def get_inputs_outputs(parts):
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


def create_subgraph(graph_module, nodes, inputs, outputs):
    '''
    Creates a module from one block of a partition.
    Returns a GraphModule that can be used like a nn.Module
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


def split_graph_constrained(graph_module, times, memories, num_parts=3):
    '''
    Splits a graph into roughly equal blocks in terms of time.
    This algorithm does not take into account the memory used or transferred.
    Unlike split_graph, it is guaranteed that every block has 1 tensor as input and 1 tensor as output.
    '''
    nodes = list(graph_module.graph.nodes)
    total_time = sum(times.get(node.name, 0) for node in nodes)
    target_time = total_time / num_parts

    parts = [[] for _ in range(num_parts)]
    needed_inputs = []

    current_part = num_parts - 1
    current_time = 0
    for node in reversed(nodes):
        parts[current_part].insert(0, node)
        current_time += times.get(node.name, 0)
        for dep in node.all_input_nodes:
            if dep.name not in needed_inputs and dep not in parts[current_part]:
                needed_inputs.append(dep.name)
        if node.name in needed_inputs:
            needed_inputs.remove(node.name)
        if current_time > target_time and len(needed_inputs) <= 1:
            if len(needed_inputs) != 0 and memories.get(needed_inputs[0], 0) == Node.NON_TENSOR: continue # non tensor edge
            current_part -= 1
            current_time = 0

    return parts

def split_graph(graph_module, times, memories, num_parts=3):
    '''
    Splits a graph into roughly equal blocks in terms of time.
    This algorithm does not take into account the memory used or transferred.
    Unlike split_graph_constrained, the number of input and output tensors of each block is NOT guaranteed.
    '''
    nodes = list(graph_module.graph.nodes)
    total_time = sum(times.get(node.name, 0) for node in nodes)
    target_time = total_time / (num_parts * .9)

    parts = []
    current_part = []
    current_time = 0

    for node in nodes:
        node_time = times.get(node.name, 0)
        mem = memories.get(node.name, 0)
        if current_time + node_time > target_time and len(parts) < num_parts - 1 and mem != Node.NON_TENSOR: # don't cut on non-tensor edges
            parts.append(current_part)
            current_part = []
            current_time = 0
        current_part.append(node)
        current_time += node_time
    parts.append(current_part)

    return parts

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


def split_graph_metis(graph, times, memories, n):
    '''
    Splits a graph into n parts using METIS.
    '''
    graph = convert_fx(graph, times, memories)
    file = write_metis(graph)
    execute_metis(file, n)
    file.close()
    parts = read_metis(graph, f'{file.name}.part.{n}')

    return parts

def partition_graph(model, n, sample, mode = "metis"):
    '''
    Splits a graph into n parts of roughly equal time.
    Different modes are available: 
    - basic: does not take into account memory, no constraint on the number of inputs/outputs
    - constrained: does not take into account memory, inputs & outputs of each block are limited to 1 tensor
    - metis (default): uses METIS to minimize both time and communication memory. No hard constraint on inputs/outputs.
    '''
    trace = torch.fx.symbolic_trace(model.cuda())
    sample = sample.cuda()
    times, memories = profile_operations(trace, sample)

    if mode == "metis":
        parts = split_graph_metis(trace, times, memories, n)        
    elif mode == "constrained":
        parts = split_graph_constrained(trace, times, memories, n)
    else:
        parts = split_graph(trace, times, memories, n)
    inputs, outputs = get_inputs_outputs(parts)
    blocks = []
    for i, p in enumerate(parts):
        graph = create_subgraph(trace, p, inputs[i], outputs[i])
        blocks.append(graph)

    return blocks, inputs, outputs

if __name__ == '__main__':
    import sys
    from utils import Timer
    sys.path.append("./")
    from models.simple import SimpleTransformer
    model = SimpleTransformer(500, 256, 6)
    sample = model.get_sample(2)
    
    trace = torch.fx.symbolic_trace(model)
    parts, inputs, outputs = partition_graph(trace, 4, sample, mode = "metis")
    
    for i,p in enumerate(parts):
        print(f'Part {i} needs inputs {inputs[i]} and has outputs {outputs[i]}.')
    
    times = []
    x = sample
    x = {i: sample for i in inputs[0]}
    for i,s in enumerate(parts):
        start = time.time()
        x = s(**x)
        times.append(time.time() - start)
        x = {y:z for y,z in x.items() if i + 1 == len(inputs) or y in inputs[i + 1]}

    print(["%.3f" % t for t in times])
    print(f'Median : {np.median(times):.3f} - Stddev : {np.std(times):.3f}')
    all_values = {x:y for x,y in x.items() if x == "view"}

    gt = model(sample)
    assert torch.allclose(x['view'], gt)
else:
    from .utils import Timer
