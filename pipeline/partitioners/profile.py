'''
Utils for operation profiling
'''

import torch
from pipeline.utils import Timer

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
    elif hasattr(x, '__iter__') and not isinstance(x, str):
        return type(x)(to_device(item, device) for item in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
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
    elif hasattr(x, '__iter__') and not isinstance(x, str):
        return sum([get_memory(v) for v in x])
    elif isinstance(x, dict):
        return sum([get_memory(v) for v in x.values()])
    return 0

def add_profiling_to_graph(graph_module, device, iters = 10):
    '''
    Torch FX graph manipulation to add profiling of time & memory to each node.

    :param graph_module: original symbolically traced module to be profiled (see torch.fx)
    :type graph_module: fx.GraphModule
    :param device: device on which the profiling will be performed
    :type device: str or torch.device
    :param iters: number of computations to perform for each operation. More iters means a more robust result, but will take more time
    :type iters: Optional[int]
        
    :return:

        - a new module that will profile the operations it runs.
        - a dictionary for time where the profiling results will be stored with the format {node_name: value}
        - a similar dictionary for output memories
        - a list of functions that restore original operations to remove profiling

    :rtype: fx.GraphModule, Dict[str, List[float]], Dict[str, List[float]], List[function() -> None]
    '''
    # list of functions to call to restore initial operations
    restore = []
    def wrap_function(node):
        original_target = node.target
        def timed_forward(*args, **kwargs):
            times = []
            for _ in range(iters):
                with Timer() as timer:
                    output = original_target(*args, **kwargs)
                times.append(timer.time())
            
            assert(node.name not in node_time)
            assert(node.name not in node_memory)
            node_time[node.name] = times
            node_memory[node.name] = get_memory(output)
            output = to_device(output, 'cpu')
            restore.append(lambda : setattr(node, "target", original_target))
            return output
        
        setattr(node, "target", timed_forward)

    def wrap_module(node):
        module = dict(graph_module.named_modules())[node.target]
        original_forward = getattr(module, "original_forward", None)
        if original_forward is None:
            original_forward = module.forward
            setattr(module, "original_forward", module.forward)
            restore.append(lambda : setattr(module, "forward", original_forward))
            next_forward = original_forward
        else:
            next_forward = getattr(module, "forward")

        def timed_forward(*args, **kwargs):
            nonlocal module
            times = []
            module = module.to(device)
            args = to_device(args, device)
            kwargs = to_device(kwargs, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if node.name in node_time:
                return original_forward(*args, **kwargs)
            else:
                for _ in range(iters):
                    with Timer() as timer:
                        output = original_forward(*args, **kwargs)
                    times.append(timer.time())
                node_time[node.name] = times
                node_memory[node.name] = get_memory(output)
                setattr(module, "forward", next_forward)

            output = to_device(output, 'cpu')
                
            module.cpu()

            return output
        
        module.forward = timed_forward
    
    def wrap_method(node):
        instance = node.args[0]
        def timed_method(*args, **kwargs):
            method = getattr(args[0], node.target)
            times = []
            args = to_device(args, device)
            kwargs = to_device(kwargs, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            for _ in range(iters):
                with Timer() as timer:
                    output = method(*args, **kwargs)
                times.append(timer.time())
            
            assert(node.name not in node_time)
            assert(node.name not in node_memory)
            node_time[node.name] = times
            node_memory[node.name] = get_memory(output)
            output = to_device(output, 'cpu')
            
            restore.append(lambda : setattr(instance, node.target, method))
            return output
        
        setattr(instance, node.target, timed_method)

    node_time = {}
    node_memory = {}

    for node in graph_module.graph.nodes:
        if node.op == 'call_function':
            wrap_function(node)

        elif node.op == 'call_module':
            wrap_module(node)

        elif node.op == 'call_method':
            wrap_method(node)

        elif node.op == 'get_attr':
            r = node.target.split('.')
            attribute = graph_module
            for lvl in r:
                attribute = getattr(attribute, lvl)

            node_memory[node.name] = get_memory(attribute)

    graph_module.recompile()
    return graph_module, node_time, node_memory, restore

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
    # Add timing to the graph
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph_module, node_time, node_memory, restore = add_profiling_to_graph(graph_module, device)

    # Execute the modified graph
    with torch.no_grad():
        graph_module(input_sample)

    for f in restore: f()

    # Fill the operations that were not timed
    for node in graph_module.graph.nodes:
        if node.name not in node_time:
            node_time[node.name] = 0
        if node.name not in node_memory:
            node_memory[node.name] = 0 # if it's not a tensor, don't cut here as it creates problems for comms

    return node_time, node_memory
