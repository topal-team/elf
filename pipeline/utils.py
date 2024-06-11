'''
Various useful classes / functions
'''
import time
import gc
import uuid
import torch

dtypes = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool
]

class TensorMetadata():
    '''
    Informations about Tensors that are sent and received in p2p communication
    [dtype, *shape]
    '''
    MAX_SIZE = 16

    @staticmethod
    def from_tensor(t):
        '''
        Creates a TensorMetadata object from its Tensor equivalent (should be used when receiving metadata via p2p)
        '''
        dtype = dtypes[int(t[0].item())]
        shape = []
        assert len(t.shape) == 1, "Metadata should only have one dimension"
        for s in t[1:]:
            s = int(s.item())
            if s == 0: break
            shape.append(s)
        
        metadata = TensorMetadata(torch.empty(0, dtype=dtype))
        metadata.shape = shape
        return metadata

    def __init__(self, t):
        self.shape = t.shape
        self.dtype = t.dtype
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    def to_tensor(self):
        '''
        Creates the Tensor representation of this metadata. Should be used when sending metadata via p2p
        '''
        t = torch.zeros(TensorMetadata.MAX_SIZE, device = self.device)
        t[0] = dtypes.index(self.dtype)
        for i, s in enumerate(self.shape):
            t[1 + i] = s
        return t
    
    def get_buffer(self, batch_size):
        '''
        Allocates a tensor with the right shape and dtype for this metadata
        '''
        buffer = torch.empty((batch_size, *self.shape), dtype = self.dtype, device = self.device)
        return buffer

    def __repr__(self):
        return f'TensorMetadata({self.shape})'


class Timer():
    def __new__(cls):
        if torch.cuda.is_available():
            return TimerGPU()
        else:
            return TimerCPU()

class TimerCPU():
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()

    def time(self):
        return (self.end - self.start) * 1000
    
class TimerGPU():
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing = True)
        self.end_event = torch.cuda.Event(enable_timing = True)
        
    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        self.end_event.synchronize()

    def time(self):
        return self.start_event.elapsed_time(self.end_event) / 1000

class activations_offloading(torch.autograd.graph.saved_tensors_hooks):
    # Singleton pattern
    _instance = None
    def __new__(class_):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_)
            class_._instance.events = {}
            class_._instance.tensors = {}
            class_._instance.stream = torch.cuda.Stream()
        return class_._instance
    
    def __init__(self):
        # When forward is done, we start sending to cpu asynchronously
        def pack_to_cpu(tensor):
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(torch.cuda.is_available() and not tensor.is_sparse),
            )

            key = uuid.uuid4()
            self.events[key] = torch.cuda.Event()
            self.tensors[key] = packed
            with torch.cuda.stream(self.stream):
                packed.copy_(tensor, non_blocking = True)
                self.events[key].record(self.stream)
                
            return (tensor.device, key, packed)

        # Ensure the data movement was finished and return the device tensor
        def unpack_from_cpu(packed):
            device, key, tensor = packed
            self.stream.synchronize()
            # If it wasn't prefetched just copy it
            if not self.events.get(key):
                print(f'Tensor was not prefetched :/ - {tensor.shape}')
                return tensor.cuda()
            
            self.events[key].synchronize()
            unpacked = self.tensors[key]
            del self.events[key]
            del self.tensors[key]
            return unpacked

        super().__init__(pack_to_cpu, unpack_from_cpu)

    # At some point we need to free memory ; this means potentially waiting for the copy to finish, so we want to do it just in time, not too early
    def wait_for_offloading(self):
        for key in self.events.keys():
            self.events[key].synchronize()

    # Copy back from device to host, asynchronously
    def prefetch(self):
        with torch.cuda.stream(self.stream):
            for key in self.events.keys():
                tensor = self.tensors[key]
                unpacked = torch.empty(
                    tensor.size(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    device=torch.cuda.current_device(),
                )
                self.tensors[key] = unpacked
                self.events[key] = torch.cuda.Event()
                unpacked.copy_(tensor, non_blocking = True)
                self.events[key].record(self.stream)
