import sys
import time
import torch
import torch.distributed as dist
import numpy as np
sys.path.append("./")
from .utils import Timer

def add_timing_to_graph(graph_module):
    def wrap_function(target):
        def timed_forward(*args, **kwargs):
            with Timer() as timer:
                output = target(*args, **kwargs)
            node_time[target.__name__] = timer.time()
            return output
        return timed_forward

    def wrap_module(module, node_name):
        original_forward = module.forward
        def timed_forward(*args, **kwargs):
            with Timer() as timer:
                output = original_forward(*args, **kwargs)
            node_time[node_name] = timer.time()
            return output
        module.forward = timed_forward
        return module
    
    def wrap_method(instance, method_name):
        def timed_method(*args, **kwargs):
            method = getattr(args[0], method_name)
            with Timer() as timer:
                output = method(*args, **kwargs)
            node_time[method_name] = timer.time()
            return output
        setattr(instance, method_name, timed_method)

    node_time = {}
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
    return graph_module, node_time, original_operations

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
    graph_module, node_time, original_ops = add_timing_to_graph(graph_module)

    # Execute the modified graph
    with torch.no_grad():
        graph_module(input_sample)

    restore_original_operations(graph_module, original_ops)
    
    return node_time

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
        if node.op == "placeholder":
            part_inputs[current_part].append(node.name)
            parts[current_part].remove(node)
        elif node.op == "output":
            part_outputs[current_part].append(node.args[0].name)
            parts[current_part].remove(node)
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
        # subgraph.output({
        #     o: env[o] for o in outputs
        # })
        if len(outputs) != 0:
            subgraph.output(env[outputs[0]])
        
    return torch.fx.GraphModule(graph_module, subgraph)

def partition_graph(model, placement, sample):
    trace = torch.fx.symbolic_trace(model.cuda())
    operation_times = profile_operations(trace, sample)

    parts, inputs, outputs = split_graph_constrained(trace, operation_times, len(placement))
    blocks = []
    for i, p in enumerate(parts):
        if placement[i] == dist.get_rank():
            blocks.append(create_subgraph(trace, p, inputs[i], outputs[i]).cuda())

    return blocks

if __name__ == '__main__':
    model = SimpleTransformer(500, 256, 12)
    sample = model.get_sample(2)
    
    trace = torch.fx.symbolic_trace(model)
    operation_times = profile_operations(trace, sample)

    parts, inputs, outputs = split_graph_constrained(trace, operation_times, 5)
    # parts, inputs, outputs = split_graph_into_parts(trace, operation_times, 5)
    for p in parts:
        print(p)
    submodules = []
    for i,p in enumerate(parts):
        print(f'Parts {p} need inputs {inputs[i]} and has outputs {outputs[i]}.')
        subgraph = create_subgraph(trace, p, inputs[i], outputs[i])
        submodules.append(subgraph)
    
    times = []
    x = sample
    # x = {i: sample for i in inputs[0]}
    # all_values = {**x}
    for i,s in enumerate(submodules):
        print(s.code)
        start = time.time()
        x = s(x)
        times.append(time.time() - start)
        # for y,z in x.items():
        #     all_values[y] = z
        # x = {y:z for y,z in all_values.items() if i + 1 < len(inputs) and y in inputs[i + 1]}

    print(times)
    print(np.median(times), np.std(times))
    # all_values = {x:y for x,y in all_values.items() if x == "view"}

    gt = model(sample)
    assert torch.allclose(x, gt)
