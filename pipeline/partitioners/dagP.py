import os
import tempfile
import subprocess
from .metis import read_metis, convert_fx

def split_graph_dagP(graph, times, memories, n):
    '''
    Split graph using dagP (https://github.com/GT-TDAlab/dagP, https://inria.hal.science/hal-02306566/document)
    Compared to METIS, this has the advantage of always creating acyclic graphs
    '''
    graph = convert_fx(graph, times, memories)
    file = write_dagP(graph)
    execute_dagP(file, n)
    file.close()
    parts = read_metis(graph, f'{file.name}.partsfile.part_{n}.seed_0.txt') # output file is the same as metis

    return parts

def write_dagP(graph):
    '''
    Writes a graph in DOT format to use dagP on.
    '''
    file = tempfile.NamedTemporaryFile("w+", dir = ".", suffix = ".dot")
    file.write('digraph compgraph {\n') # don't forget the \n because dagP dot reader is sensitive ;)
    for node in graph.values():
        file.write(node.to_dot_line())
    for node in graph.values():
        file.write(node.to_dot_edges())
    file.write('}')
    return file

def execute_dagP(file, n):
    '''
    Execute dagP on a file and clean all unnecessary files
    '''
    file.flush()  # Ensure all data is written before executing
    file.seek(0)  # Reset file pointer to the beginning for reading in subprocess
    subprocess.run(["rMLGP", file.name, str(n), "--obj", "1", "--write_parts", "1", "--print", "0"], stdout = subprocess.DEVNULL)
    os.remove(f'{file.name}.bin')
    os.remove(f'{file.name}.nodemappings')
    os.remove(f'{file.name}.partitioned.part_{n}.seed_0.dot')

# No read_dagP because the output is the same as METIS. Use read_metis instead.