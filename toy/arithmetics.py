import sys
import torch
import torch.nn.functional as F

dtype = torch.bfloat16
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

op = sys.argv[1] if len(sys.argv) > 1 else "maxpool"

if op == "linear":
    N = 512
    M = 4096
    K = 1024

    A = torch.rand(N, K, device='cuda', dtype = dtype)
    B = torch.rand(K, M, device='cuda', dtype = dtype)
    D = torch.rand(N, M, device='cuda', dtype = dtype)
    C = torch.zeros(N, M, device='cuda', dtype = dtype)

    num_operations = 2 * N * M * K + (N * M)

    # Compute the amount of memory transferred
    # Assume they all have the same dtype
    memory_transferred = (A.numel() + B.numel() + C.numel() + D.numel()) * A.element_size()

    # Warm-up
    C = torch.mm(A, B)

    start_event.record()
    C = torch.mm(A, B) + D
    end_event.record()
    
elif op == "maxpool":
    batch_size = 256
    channels = 64
    height = 224
    width = 224

    input_tensor = torch.rand(batch_size, channels, height, width, device='cuda', dtype=dtype)
    output_tensor = torch.zeros(batch_size, channels, height, width, device='cuda', dtype=dtype)

    kernel_size = 3
    stride = 1
    padding = 1

    num_operations = batch_size * channels * height * width * (kernel_size * kernel_size - 1)

    # Compute the amount of memory transferred
    input_memory = input_tensor.numel() * input_tensor.element_size()
    output_memory = output_tensor.numel() * output_tensor.element_size()
    memory_transferred = input_memory + output_memory

    # Warm-up
    output_tensor = F.max_pool2d(input_tensor, kernel_size, stride, padding)

    start_event.record()
    output_tensor = F.max_pool2d(input_tensor, kernel_size, stride, padding)
    end_event.record()

elif op == "relu":
    batch_size = 256
    channels = 64
    height = 224
    width = 224

    input_tensor = torch.rand(batch_size, channels, height, width, device='cuda', dtype=dtype)
    output_tensor = torch.zeros(batch_size, channels, height, width, device='cuda', dtype=dtype)

    num_operations = batch_size * channels * height * width

    # Compute the amount of memory transferred
    input_memory = input_tensor.numel() * input_tensor.element_size()
    output_memory = output_tensor.numel() * output_tensor.element_size()
    memory_transferred = input_memory + output_memory

    # Warm-up
    output_tensor = F.relu(input_tensor)

    start_event.record()
    output_tensor = F.relu(input_tensor)
    end_event.record()
    
elif op == "layernorm":
    batch_size = 256
    sequence_length = 128
    feature_dim = 2048

    # Initialize the input tensor
    input_tensor = torch.rand(batch_size, sequence_length, feature_dim, device='cuda', dtype=dtype)
    output_tensor = torch.zeros_like(input_tensor)

    num_elements = input_tensor.numel()
    num_operations = 6 * num_elements

    input_memory = input_tensor.numel() * input_tensor.element_size()
    output_memory = output_tensor.numel() * output_tensor.element_size()
    memory_transferred = input_memory + output_memory

    # Define LayerNorm parameters
    eps = 1e-5

    output_tensor = F.layer_norm(input_tensor, (feature_dim,), eps=eps)

    start_event.record()
    output_tensor = F.layer_norm(input_tensor, (feature_dim,), eps=eps)
    end_event.record()

elif op == "conv":
    # Define input tensor size (e.g., batch_size x channels x height x width)
    batch_size = 256
    in_channels = 64
    out_channels = 128
    height = 224
    width = 224

    # Define convolution parameters
    kernel_size = 3
    stride = 1
    padding = 1

    # Initialize the input tensor
    input_tensor = torch.rand(batch_size, in_channels, height, width, device='cuda', dtype=dtype)
    conv_weight = torch.rand(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=dtype)
    output_tensor = torch.zeros(batch_size, out_channels, height, width, device = 'cuda', dtype=dtype)

    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    num_operations = out_channels * output_height * output_width * (kernel_size * kernel_size * in_channels * 2)

    input_memory = input_tensor.numel() * input_tensor.element_size()
    output_memory = output_tensor.numel() * output_tensor.element_size()
    weight_memory = conv_weight.numel() * conv_weight.element_size()
    memory_transferred = input_memory + output_memory + weight_memory
    
    # Warm-up
    output_tensor = F.conv2d(input_tensor, conv_weight, stride=stride, padding=padding)

    start_event.record()
    output_tensor = F.conv2d(input_tensor, conv_weight, stride=stride, padding=padding)
    end_event.record()
    
torch.cuda.synchronize()
time_taken = start_event.elapsed_time(end_event) / 1000  # convert to seconds

arithmetic_intensity = num_operations / memory_transferred

achieved_flops = num_operations / time_taken

print(f"Time taken for operations: {time_taken:.6f} seconds")
print(f"Number of operations: {num_operations}")
print(f"Memory transferred: {memory_transferred / (1024**2):.6f} MB")
print(f"Arithmetic intensity: {arithmetic_intensity:.6f} operations/byte")
print(f"Achieved FLOPS: {achieved_flops / 1e12:.6f} TFLOPS")

# Theoretical peak values (replace these with your GPU's specs)
peak_flops = 15.7 * 1e12
peak_bandwidth = 900 * 1e9

# Determine the limiting factor
if arithmetic_intensity < (peak_flops / peak_bandwidth):
    print("The operation is memory-bound.")
else:
    print("The operation is compute-bound.")
