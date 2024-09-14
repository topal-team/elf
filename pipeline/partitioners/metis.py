'''
Utils to partition a computation graph using METIS
'''

import os
import shutil
import random
import tempfile
import subprocess
import numpy as np

# Some edges should not be part of the cut; for instance values that are not torch.Tensor are a problem to send/recv for pipelining.
# To avoid that we put a soft constraint by giving them a huge communication cost.
NON_TENSOR = 2 << 20

class Node:
    '''
    Custom representation of graphs to manipulate METIS inputs/outputs easily
    '''
    def __init__(self, node, time, mem, idx):
        '''
        :param node: original node
        :type node: fx.Node
        :param time: time obtained by profiling this node
        :type time: float
        :param mem: memory size of this node's output
        :type mem: float
        :param idx: index of the node in the (topological sort of) the graph
        :type idx: int
        '''
        self.node = node
        self.time = time
        self.idx = idx
        self.mem = mem
        # TODO: change to children/parents
        self.edges_in = []
        self.edges_out = []

    def to_metis_line(self):
        '''
        s w1 w2 ... wncon v1 e1 v2 e2 ... vk ek
        where s is the size of the vertex, w1, w2, . . . , wncon are the ncon vertex weights associated with this vertex, v1, . . . , vk
        are the vertices adjacent to this vertex, and e1, . . . , ek are the weights of these edges (undirected).
        http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/manual.pdf - Section 4.1.1

        :return: representation of this node as expected by METIS
        :rtype: str
        '''
        line = f'{self.mem} {self.time}'

        for v in self.edges_in:
            line += f' {v}'
        for v,_ in self.edges_out:
            line += f' {v}'
        return line + f"% {self.idx} - {self.node.name}"
    
    def to_dot_line(self):
        '''
        Dot representation of node
        https://graphviz.org/doc/info/lang.html

        :return: representation of this node in dot format
        :rtype: str
        '''
        dot = f'{self.idx} [weight={self.time}];\n'
        return dot
    
    def to_dot_edges(self):
        '''
        Dot representation of all edges (directed)

        :return: representation of all outgoing edges for this node, in dot format
        :rtype: str
        '''
        dot = ''
        for v,e in self.edges_out:
            dot += f'{self.idx}->{v} [weight={e}];\n'
        return dot
    
    def __repr__(self):
        return repr(self.node)

def convert_fx(module, times, memories):
    '''
    Converts a torch FX graph into a custom format using the Node class
    This format includes informations about the time & memory profiling, and edges between each nodes

    :param module: symbolic trace of a torch model (see torch.fx)
    :type module: fx.GraphModule
    :param times: information about the weights, i.e. the profiled execution time, of each node
    :type times: Dict[str, float]
    :param memories: information about the weight of the outgoing edge for each node, i.e. the memory size of the output value
    :type memories: Dict[str, float]

    :return: custom representation of every node by their name
    :rtype: Dict[str, Node]
    '''
    nodes = list(module.graph.nodes)
    graph = {}
    indices = {}

    times = {k: np.median(v) for k,v in times.items()}
    memories = {k: np.median(v) for k,v in memories.items()}
    
    to_weight = lambda x : max(int(x), 1)

    # Map into [1, 100] to have consistent times
    min_time = min(filter(lambda x : x != 0, times.values()))
    max_time = max(times.values())
    time_range = max_time - min_time
    scaled_times = {name: to_weight(1 + 99 * ((time - min_time) / time_range)) for name, time in times.items()}
    times = scaled_times

    # Map memories into [1, 100] to have consistent memory sizes
    tensor_memories = list(filter(lambda x : x != 0, memories.values()))
    min_memory = min(tensor_memories)
    max_memory = max(tensor_memories)
    memory_range = max_memory - min_memory
    scaled_memories = {name: to_weight(1 + 99 * ((memory - min_memory) / memory_range)) if memory != 0 else NON_TENSOR for name, memory in memories.items()}
    memories = scaled_memories
    
    for i, node in enumerate(nodes):
        # Indices of METIS are 1-based :(
        graph[node.name] = Node(node, times[node.name], memories[node.name], i + 1)
        indices[node.name] = i + 1
        for dep in node.all_input_nodes:
            weight = memories[dep.name]
            graph[dep.name].edges_out.append((indices[node.name], weight))
            graph[node.name].edges_in.append((indices[dep.name]))

    return graph

def split_graph_metis(graph, times, memories, n):
    '''
    Splits a graph into n parts using METIS.

    :param graph: symbolic trace of the module to partition (see torch.fx)
    :type graph: fx.GraphModule
    :param times: information about the weights, i.e. the profiled execution time, of each node
    :type times: Dict[str, float]
    :param memories: information about the weight of the outgoing edge for each node, i.e. the memory size of the output value
    :type memories: Dict[str, float]
    :param n: number of partitions to create
    :type n: Optional[int]

    :return: ``n`` lists of nodes corresponding to each part
    :rtype: List[List[fx.Node]]
    '''
    assert shutil.which("gpmetis"), "METIS chosen as partition strategy, but can't find it. Please make sure it is installed and findable in the PATH."
    graph = convert_fx(graph, times, memories)
    file = write_metis(graph)
    execute_metis(file, n)
    file.close()
    parts = read_metis(graph, f'{file.name}.part.{n}')

    return parts

def write_metis(graph):
    '''
    Dumps the info about a custom graph into a file in METIS format

    :param graph: custom representation of every node of a computational graph, by name
    :type graph: Dict[str, Node]

    :rtype: file descriptor where the info was written
    :type: File
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

    :param graph: custom representation of every node of a computational graph, by name
    :type graph: Dict[str, Node]
    :param file: name of the file where METIS wrote its output
    :type file: str

    :return: lists of nodes corresponding to each part of the partition obtained
    :rtype: List[List[fx.Node]]
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
        if i not in mapping:
            if (l < len(lines) - 1 and i == lines[l + 1]):
                mapping.append(i)
                parts.append([])
            i = -1
        else:
            i = len(mapping) - 1
        parts[i].append(nodes[n].node)
        assert nodes[n].idx == n + 1
        n += 1

    f.close()
    os.remove(f.name)
    return parts

def execute_metis(file, n):
    '''
    Runs METIS on a file

    :param file: file descriptor of the file where the info of the graph was written
    :type file: File
    :param n: number of parts to create
    :type n: int
    '''
    file.flush()  # Ensure all data is written before executing
    file.seek(0)  # Reset file pointer to the beginning for reading in subprocess
    subprocess.run(["gpmetis", file.name, str(n), "-objtype=vol", "-contig", "-ufactor=500", "-ncuts=2000", "-niter=2000"], stdout=subprocess.DEVNULL)  # Execute gpmetis

def export_partition_to_dot(parts):
    text = "digraph partition {\n"
    for i, p in enumerate(parts):
        color = '#'+ ''.join([random.choice('0123456789abcdef') for _ in range(6)])
        text += f"\tsubgraph cluster_{i} {{\n"
        text += f"bgcolor=\"{color}\";\n"
        for n in p:
            text += f'{n.idx} [weight={n.time}];\n'
        text += "}"
    for p in parts:
        for n in p:
            text += n.to_dot_edges()
    text += "}"
    with open("partition.dot", "w+") as f:
        f.write(text)
