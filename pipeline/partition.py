import time
import tempfile
import numpy as np
import subprocess
import torch
import torch.distributed as dist

class Node:
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
        line = f'{int(self.time * 1e1)}'
        for v,e in self.edges:
            line += f' {v} {max(int(e / 1e3), 1)}'
        return line
    
    def __repr__(self):
        return repr(self.node)
    
def convert_fx(module, times, memories):
    nodes = list(module.graph.nodes)
    graph = {}
    indices = {}
    for i, node in enumerate(nodes):
        graph[node.name] = Node(node, times[node.name], i + 1)
        indices[node.name] = i + 1
        for dep in node.all_input_nodes:
            graph[dep.name].edges.append((indices[node.name], memories[dep.name]))
            graph[node.name].edges.append((indices[dep.name], memories[dep.name]))

    return graph

def write_metis(graph):
    file = tempfile.NamedTemporaryFile("w+")
    n = len(graph)
    m = sum(map(lambda n : len(n.edges), list(graph.values()))) // 2 # edges are counted twice
    fmt = '011'
    ncon = 1
    file.write(f'{n} {m} {fmt} {ncon}\n')
    for node in graph.values():
        file.write(node.to_line() + "\n")
    return file

def execute_metis(file, n):
    file.flush()  # Ensure all data is written before executing
    file.seek(0)  # Reset file pointer to the beginning for reading in subprocess
    subprocess.run(["gpmetis", file.name, str(n), "-objtype=vol", "-contig"])  # Execute gpmetis with the file and partition into 5 parts

def read_metis(graph, file):
    f = open(file, "r")
    nodes = list(graph.values())
    n = 0
    mapping = []
    parts = []
    while l := f.readline():
        i = int(l)
        if i in mapping:
            i = mapping.index(i)
        else:
            mapping.append(i)
            i = len(mapping) - 1
            parts.append([])

        parts[i].append(nodes[n].node)
        n += 1
    return parts

def fx_from_metis(graph, file):
    parts = []
    nodes = list(graph.values())
    with open(file, "r") as f:
        i = 0
        while line := f.readline():
            p = int(line) - 1
            parts[p].append(nodes[i].node)
            i += 1
    return parts

def add_profiling_to_graph(graph_module):
    def wrap_function(target):
        def timed_forward(*args, **kwargs):
            with Timer() as timer:
                for _ in range(10):
                    output = target(*args, **kwargs)
            node_time[target.__name__] = timer.time() / 10
            node_memory[target.__name__] = output.numel() * output.element_size()
            return output
        return timed_forward

    def wrap_module(module, node_name):
        original_forward = module.forward
        def timed_forward(*args, **kwargs):
            with Timer() as timer:
                for _ in range(10):
                    output = original_forward(*args, **kwargs)
            node_time[node_name] = timer.time() / 10
            node_memory[node_name] = output.numel() * output.element_size()
            return output
        module.forward = timed_forward
        return module
    
    def wrap_method(instance, method_name):
        def timed_method(*args, **kwargs):
            method = getattr(args[0], method_name)
            with Timer as timer:
                output = method(*args, **kwargs)
            node_time[method_name] = timer.time()
            node_memory[method_name] = output.numel() * output.element_size()
            return output
        setattr(instance, method_name, timed_method)

    node_time = {}
    node_memory = {}
    original_operations = {}
    for node in graph_module.graph.nodes:
        if node.op == 'call_function':
            original_operations[node.name] = node.target
            node.target = wrap_function(node.target)
        elif node.op == 'call_module':
            submodule = dict(graph_module.named_modules())[node.target]
            original_forward = submodule.forward
            wrapped_submodule = wrap_module(submodule, node.name)
            original_operations[node.name] = original_forward
            setattr(graph_module, node.target, wrapped_submodule)
        # elif node.op == 'call_method':
        #     instance = node.args[0]
        #     method_name = node.target
        #     original_method = wrap_method(instance, method_name, node.name)
        #     original_operations[node] = (instance, method_name, original_method)


    graph_module.recompile()
    return graph_module, node_time, node_memory, original_operations

def restore_original_operations(graph_module, original_operations):
    for node in graph_module.graph.nodes:
        if node.name in original_operations:
            if node.op == 'call_function' or node.op == 'call_method':
                node.target = original_operations[node.name]
            elif node.op == 'call_module':
                submodule = dict(graph_module.named_modules())[node.target]
                submodule.forward = original_operations[node.name]

    graph_module.recompile()

def profile_operations(graph_module, input_sample):
    # Add timing to the graph
    graph_module, node_time, node_memory, original_ops = add_profiling_to_graph(graph_module)

    # Execute the modified graph
    with torch.no_grad():
        graph_module(input_sample)

    restore_original_operations(graph_module, original_ops)

    for node in graph_module.graph.nodes:
        if node.name not in node_time:
            node_time[node.name] = 0
        if node.name not in node_memory:
            node_memory[node.name] = 0

    return node_time, node_memory

def split_graph_constrained(graph_module, node_times, num_parts=3):
    nodes = list(graph_module.graph.nodes)
    total_time = sum(node_times.get(node.name, 0) for node in nodes)
    target_time = total_time / num_parts

    parts = [[] for _ in range(num_parts)]
    needed_inputs = []

    part_inputs = {i: [] for i in range(len(parts))}
    part_outputs = {i: [] for i in range(len(parts))}

    current_part = num_parts - 1
    current_time = 0
    for node in reversed(nodes):
        parts[current_part].insert(0, node)
        current_time += node_times.get(node.name, 0)
        for dep in node.all_input_nodes:
            if dep.name not in needed_inputs and dep not in parts[current_part]:
                needed_inputs.append(dep.name)
        if node.name in needed_inputs:
            needed_inputs.remove(node.name)
        if current_time > target_time and len(needed_inputs) <= 1:
            part_inputs[current_part] = needed_inputs.copy()
            part_outputs[current_part - 1] = needed_inputs.copy()
            current_part -= 1
            current_time = 0

    return parts, part_inputs, part_outputs

def split_graph_into_parts(graph_module, node_times, num_parts=3):
    nodes = list(graph_module.graph.nodes)
    total_time = sum(node_times.get(node.name, 0) for node in nodes)
    target_time = total_time / (num_parts * .9)

    parts = []
    current_part = []
    current_time = 0

    for node in nodes:
        node_time = node_times.get(node.name, 0)
        if current_time + node_time > target_time and len(parts) < num_parts - 1:
            parts.append(current_part)
            current_part = []
            current_time = 0
        current_part.append(node)
        current_time += node_time
    parts.append(current_part)

    # Dictionary to keep track of nodes that are outputs for each part
    part_inputs = {i: [] for i in range(len(parts))}
    part_outputs = {i: [] for i in range(len(parts))}

    # Iterate over each part in reversed order
    index = len(parts) - 1
    for part in reversed(parts):
        # Set to store all nodes in the current part
        current_part_nodes = set(node.name for node in part)
        # Iterate over each node in the current part
        for node in part:
            if node.op == "placeholder":
                part_inputs[index].append(node.name)
                part.remove(node)
            elif node.op == "output":
                part_outputs[index].append(node.args[0].name)
                part.remove(node)
            # Check dependencies of the node
            for dep_node in node.all_input_nodes:
                # If dependency node is not in the current part, mark the dependency node as output
                if dep_node.name not in current_part_nodes:
                    part_inputs[index].append(dep_node.name)
                    # Find the original part index of the dependency node
                    for original_index, original_part in enumerate(parts[:index]):
                        if dep_node in original_part:
                            part_outputs[original_index].append(dep_node.name)
                            break

        index -= 1

    part_inputs =  {i: set(j) for i,j in part_inputs.items()}
    part_outputs =  {i: set(j) for i,j in part_outputs.items()}
    return parts, part_inputs, part_outputs

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
            print(f'Processing node {node} - needs inputs {node.all_input_nodes}')
            for dep in node.all_input_nodes:
                if dep not in part:
                    print(f'Dep {dep} not in part {part} !')
                    inputs[i].add(dep.name)
                    outputs[find_idx(dep)].add(dep.name)
    
    return inputs, outputs # remove dupes


def create_subgraph(graph_module, nodes, inputs, outputs):
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
        # if len(outputs) != 0:
        #     subgraph.output(env[outputs[0]])
        
    return torch.fx.GraphModule(graph_module, subgraph)

def split_graph_metis(graph, times, memories, n):
        graph = convert_fx(graph, times, memories)
        file = write_metis(graph)
        execute_metis(file, n)
        file.close()
        parts = read_metis(graph, f'{file.name}.part.{n}')
        inputs, outputs = get_inputs_outputs(parts)

        return parts, inputs, outputs

def partition_graph(model, placement, sample, mode = "metis"):
    trace = torch.fx.symbolic_trace(model.cuda())
    operation_times, operation_memories = profile_operations(trace, sample)

    if mode == "metis":
        parts, inputs, outputs = split_graph_metis(trace, operation_times, operation_memories, len(placement))        
    elif mode == "constrained":
        parts, inputs, outputs = split_graph_constrained(trace, operation_times, len(placement))
    else:
        parts, inputs, outputs = split_graph_into_parts(trace, operation_times, len(placement))
    blocks = []
    for i, p in enumerate(parts):
        if placement[i] == dist.get_rank():
            blocks.append(create_subgraph(trace, p, inputs[i], outputs[i]).cuda())

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
    
    submodules = []
    for i,p in enumerate(parts):
        print(f'Parts {p} need inputs {inputs[i]} and has outputs {outputs[i]}.')
        subgraph = create_subgraph(trace, p, inputs[i], outputs[i])
        submodules.append(subgraph)
    
    times = []
    x = sample
    x = {i: sample for i in inputs[0]}
    all_values = {**x}
    for i,s in enumerate(submodules):
        print(s.code)
        start = time.time()
        x = s(**x)
        times.append(time.time() - start)
        for y,z in x.items():
            all_values[y] = z
        x = {y:z for y,z in all_values.items() if i + 1 == len(inputs) or y in inputs[i + 1]}

    print(times)
    print(np.median(times), np.std(times))
    all_values = {x:y for x,y in all_values.items() if x == "view"}

    gt = model(sample)
    assert torch.allclose(x['view'], gt)
else:
    from .utils import Timer