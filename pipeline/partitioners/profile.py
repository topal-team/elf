'''
Utils for operation profiling
'''

import torch
from pipeline.utils import Timer

class Profiler(torch.fx.Interpreter):
    def __init__(self, *args, **kwargs):
        self.times = {}
        self.memories = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(Profiler, self).__init__(*args, **kwargs)
    
    def run_node(self, node):
        # Move all inputs to the specified device
        args = [to_device(arg, self.device) for arg in node.args]
        kwargs = {k: to_device(v, self.device) for k, v in node.kwargs.items()}
        node.args = tuple(args)
        node.kwargs = kwargs
        if torch.cuda.is_available(): torch.cuda.synchronize()

        with Timer() as timer:
            result = super().run_node(node)

        # Move all modules back to CPU
        for arg in args:
            if isinstance(arg, torch.nn.Module):
                arg.to('cpu')
        for _, kwarg in kwargs.items():
            if isinstance(kwarg, torch.nn.Module):
                kwarg.to('cpu')
        
        self.times[node.name] = timer.time()
        self.memories[node.name] = get_memory(result)
        
        return result

def to_device(x, device):
    '''
    Moves ``x`` to the specified device if it is a tensor
    If ``x`` is an iterable or dictionary, recursively moves every tensor contained in it to the device too
    Otherwise does not do anything

    :param x: object to move to device memory
    :type x: Any
    :param device: destination
    :type device: str or torch.device

    :return: moved objects
    :rtype: same as ``x``
    '''
    if isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Module):
        return x.to(device, non_blocking = True)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif hasattr(x, '__iter__') and not isinstance(x, (str, torch.fx.proxy.Proxy)):
        return type(x)(to_device(item, device) for item in x)
    return x

def get_memory(x):
    '''
    Estimates memory used by ``x`` if it is a tensor
    If ``x`` is an iterable or dictionary, recursively count memory for every tensor contained in it

    :param x: object to estimate
    :type x: Any

    :return: memory estimation in bytes
    :rtype: float
    '''
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    elif isinstance(x, dict):
        return sum([get_memory(v) for v in x.values()])
    elif hasattr(x, '__iter__') and not isinstance(x, (str, torch.fx.proxy.Proxy)):
        return sum([get_memory(v) for v in x])
    return 0

def profile_operations(graph_module, input_sample):
    '''
    Get time and memory for each node of a traced module, when running forward on a sample.

    :param graph_module: original symbolically traced module to be profiled (see torch.fx)
    :type graph_module: fx.GraphModule
    :param input_sample: example of inputs to feed to the model
    :type input_sample: Tensor

    :return: 2 dicts, one for time and one for memory used, respectively. Each one has the format {node_name: value}
    :rtype: fx.GraphModule, Dict[str, List[float]], Dict[str, List[float]]
    '''    
    profiler = Profiler(graph_module)
    profiler.run(input_sample)

    return profiler.times, profiler.memories
