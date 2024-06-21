import gc
import torch
import numpy as np
import torch.nn as nn
from pipeline.utils import Timer

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
    if torch.cuda.is_available():
        graph_module = DynamicSplitForward(graph_module) # If we can't fit everything in memory, we split the module
        graph_module.find_split(input_sample)

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

class DynamicSplitForward(nn.Module):
    def __init__(self, module, initial_splits=1):
        super(DynamicSplitForward, self).__init__()
        self.module = torch.fx.symbolic_trace(module)
        self.initial_splits = initial_splits
        self.module_parts = None
        self._split_module(initial_splits)
    
    def _split_module(self, num_splits):
        nodes = list(self.module.graph.nodes)
        total_nodes = len(nodes)
        split_size = (total_nodes + num_splits - 1) // num_splits  # Ceiling division
        
        module_parts = []
        for i in range(num_splits):
            start = i * split_size
            end = min((i + 1) * split_size, total_nodes)
            
            subgraph = self.module.graph.create_subgraph(nodes[start:end])
            submodule = torch.fx.GraphModule(self.module, subgraph)
            module_parts.append(submodule)
        
        self.module_parts = nn.ModuleList(module_parts)
    
    def forward(self, x):
        with torch.no_grad():
            for part in self.module_parts:
                part = part.to('cuda')
                torch.cuda.synchronize()
                x = part(x)
                part = part.to('cpu')
            return x

    def find_split(self, x):
        # Initial run with the given number of splits
        current_splits = self.initial_splits
        while True:
            try:
                self.forward(x)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    current_splits *= 2
                    self._split_module(current_splits)
                    print(f"Increasing splits to {current_splits} due to memory error.")
                    x = x.detach()
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.find_split(x)
                else:
                    raise e