import os
import tempfile
import subprocess
from .profile import NON_TENSOR

class Node:
    '''
    Custom representation of graphs to manipulate METIS inputs/outputs easily
    '''

    def __init__(self, node, time, memory, idx):
        self.node = node
        self.time = time
        self.mem = memory
        self.idx = idx
        self.edges_in = []
        self.edges_out = []

    def to_metis_line(self):
        '''
        s w1 w2 ... wncon v1 e1 v2 e2 ... vk ek
        where s is the size of the vertex, w1, w2, . . . , wncon are the ncon vertex weights associated with this vertex, v1, . . . , vk
        are the vertices adjacent to this vertex, and e1, . . . , ek are the weights of these edges (undirected).
        http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf - Section 4.1.1
        '''
        line = f'{self.mem} {self.time}'
        for v,e in self.edges_in:
            line += f' {v}'
        for v,e in self.edges_out:
            line += f' {v}'
        return line
    
    def to_dot_line(self):
        '''
        Dot representation of node
        https://graphviz.org/doc/info/lang.html
        '''
        dot = f'{self.idx} [weight={self.time}];\n'
        return dot
    
    def to_dot_edges(self):
        '''
        Dot representation of all edges (directed)
        '''
        dot = ''
        for v,e in self.edges_out:
            dot += f'{self.idx}->{v} [weight={e}];\n'
        return dot
    
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

    to_weight = lambda x : max(int(x), 1)

    # Map into [1, 100] to have consistent times
    min_time = min(filter(lambda x : x != 0, times.values()))
    max_time = max(times.values())
    time_range = max_time - min_time
    scaled_times = {name: to_weight(1 + 999 * ((time - min_time) / time_range)) for name, time in times.items()}
    times = scaled_times

    # Map memories into [1, 100] to have consistent memory sizes
    tensor_memories = list(filter(lambda x : x != NON_TENSOR, memories.values()))
    min_memory = min(tensor_memories)
    max_memory = max(tensor_memories)
    memory_range = max_memory - min_memory
    scaled_memories = {name: to_weight(1 + 999 * ((memory - min_memory) / memory_range)) if memory != NON_TENSOR else 10000 for name, memory in memories.items()}
    memories = scaled_memories

    for i, node in enumerate(nodes):
        # Indices of METIS are 1-based :(
        graph[node.name] = Node(node, times[node.name], memories[node.name], i + 1)
        indices[node.name] = i + 1
        for dep in node.all_input_nodes:
            weight = scaled_memories[dep.name]
            graph[dep.name].edges_out.append((indices[node.name], weight))
            graph[node.name].edges_in.append((indices[dep.name], weight))

    return graph

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

def write_metis(graph):
    '''
    Dumps the info about a custom graph into a file in METIS format
    '''
    file = tempfile.NamedTemporaryFile("w+", dir = ".")
    n = len(graph)
    m = sum(map(lambda n : len(n.edges_in), list(graph.values())))
    fmt = '110'
    ncon = 1
    file.write(f'{n} {m} {fmt} {ncon}\n')
    for node in graph.values():
        file.write(node.to_metis_line() + "\n")
    return file

def read_metis(graph, file):
    '''
    Reads the output of Graph Partitioning METIS and breaks a custom graph accordingly.
    '''

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
        assert nodes[n].idx == n + 1
        n += 1

    f.close()
    os.remove(f.name)
    return parts

    '''
    f = open(file, "r")
    
    mapping = []
    nodes = list(graph.values())
    lines = list(map(int, f.readlines()))
    parts = []
    n = 0
    
    for l in range(len(lines)):
        i = lines[l]
        # WE NEED TO HAVE CONTIGUOUS PARTITIONS
        # METIS does not enforce that, so we have to fix it by ignoring some attributions
        if i not in mapping and (l < len(lines) - 1 and i == lines[l + 1]):
            mapping.append(i)
            parts.append([])
        i = len(mapping) - 1            
        parts[i].append(nodes[n].node)
        assert nodes[n].idx == n + 1
        n += 1

    f.close()
    os.remove(f.name)
    return parts
    '''

def execute_metis(file, n):
    '''
    Runs METIS on a file
    '''
    file.flush()  # Ensure all data is written before executing
    file.seek(0)  # Reset file pointer to the beginning for reading in subprocess
    subprocess.run(["gpmetis", file.name, str(n), "-objtype=vol", "-contig", "-minconn", "-ufactor=300", "-ncuts=2000"], stdout=subprocess.DEVNULL)  # Execute gpmetis
