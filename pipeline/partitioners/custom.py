import torch

def split_graph(graph_module, times, memories, num_parts=3):
    '''
    Naively splits a graph into roughly equal blocks in terms of time.
    This algorithm does not take into account the memory used or transferred.
    Unlike split_graph_constrained, the number of input and output tensors of each block is NOT guaranteed.
    '''
    nodes = list(graph_module.graph.nodes)
    total_time = sum(times.get(node.name, 0) for node in nodes)
    target_time = total_time / num_parts

    parts = []
    current_part = []
    current_time = 0

    for node in nodes:
        node_time = times.get(node.name, 0)
        if current_time + node_time > target_time * (len(parts) + 1) and len(parts) < num_parts:
            parts.append(current_part)
            current_part = []
        current_part.append(node)
        current_time += node_time
    parts.append(current_part)

    return parts

def split_graph_constrained(graph_module, times, memories, num_parts=3):
    '''
    Naively splits a graph into roughly equal blocks in terms of time.
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
            current_part -= 1
            current_time = 0
            
    return parts
