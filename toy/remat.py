import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CheckpointedModule(nn.Module):
    def __init__(self, module):
        super(CheckpointedModule, self).__init__()
        self.module = module

    def forward(self, *inputs):
        self.saved_tensors = inputs
        return CheckpointFunction.apply(self.module, *inputs)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *inputs):
        ctx.module = module
        with torch.no_grad():
            outputs = module(*inputs)
        ctx.save_for_backward(*inputs)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        with torch.enable_grad():
            outputs = ctx.module(*inputs)
        grad_inputs = torch.autograd.grad(outputs, inputs, grad_outputs)
        return (None, *grad_inputs)

class SimpleCNN(nn.Module):
    def __init__(self, remat = False):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)
        if remat:
            self.conv1 = CheckpointedModule(self.conv1)
            self.conv2 = CheckpointedModule(self.conv2)
            self.conv3 = CheckpointedModule(self.conv3)
            self.maxpool = CheckpointedModule(self.maxpool)
            self.fc1 = CheckpointedModule(self.fc1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x), inplace = True)
        x = self.maxpool(x)
        x = torch.nn.functional.relu(self.conv2(x), inplace = True)
        x = self.maxpool(x)
        x = torch.nn.functional.relu(self.conv3(x), inplace = True)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x), inplace = True)
        x = self.fc2(x)
        return x

class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim, output_dim, remat = False):
        super(SimpleAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.head = nn.Linear(hidden_dim, output_dim)
        
        if remat:
            self.query = CheckpointedModule(self.query)
            self.key = CheckpointedModule(self.key)
            self.value = CheckpointedModule(self.value)
            self.softmax = CheckpointedModule(self.softmax)

    def forward(self, inputs):
        # Linear projections
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)

        # Compute the weighted sum of values
        context = torch.matmul(attention_weights, V)

        # Prediction
        y = self.head(context)

        return y.view(-1, y.size(-1))
    
def run(data, update = True):
    model, inputs, labels, optimizer = data
    
    # Zero the parameter gradients
    if update: optimizer.zero_grad()

    # Forward + backward + optimize
    outputs = model(inputs)
    loss = nn.functional.cross_entropy(outputs, labels)
    loss.backward()
    if update: optimizer.step()

def check_memory(model, model_type):
    optimizer = torch.optim.Adam(model.parameters())
    batch_size = 128
    max_batch_size = 0
    while True:
        try:
            # Generate random input data and labels
            inputs, labels = generate_inputs(model_type, batch_size)
            torch.cuda.synchronize()
            run((model, inputs, labels, optimizer), update = True)
            # Successfully completed this batch size, increase it
            max_batch_size = batch_size
            print(f"Batch size {batch_size} succeeded")
            batch_size *= 2

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"Out of memory at batch size {batch_size}")
                break
            else:
                raise e

    print(f"Maximum batch size without OOM: {max_batch_size}\n")
    return max_batch_size

def generate_inputs(model_type, batch_size):
    if model_type == 'attention':
        inputs = torch.randn((batch_size, 256, 512)).cuda()
        labels = torch.randint(0, 500, (batch_size, 256,)).cuda().view(-1)
    elif model_type == 'cnn':
        inputs = torch.randn(batch_size, 3, 224, 224).cuda()
        labels = torch.randint(0, 10, (batch_size,)).cuda()

    return inputs, labels

if __name__ == "__main__":    
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'attention'
    if model_type == 'attention':
        model = SimpleAttention(512, 500).cuda()
    elif model_type == 'cnn':
        model = SimpleCNN().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    batch_size = check_memory(model, model_type)

    inputs, labels = generate_inputs(model_type, batch_size)
    
    iters = 10

    # Warmup
    for _ in range(3):
        run((model, inputs, labels, optimizer), update = False)

    torch.cuda.cudart().cudaProfilerStart()
    
    times = []
    for _ in range(iters):
        start_event = torch.cuda.Event(enable_timing = True)
        end_event = torch.cuda.Event(enable_timing = True)
        start_event.record()
        
        run((model, inputs, labels, optimizer), update = False)
        run((model, inputs, labels, optimizer), update = True)
        
        end_event.record()
        end_event.synchronize()
        time_taken = start_event.elapsed_time(end_event) / 1000  # convert to seconds
        times.append(time_taken)

    median = sorted(times)[iters // 2]
    print(f'Without remat, 2 iterations w/ batch size = {batch_size} : {median:.2f}s.\nThroughput = {(batch_size * 2) // median} img/s.\n')

    batch_size *= 2
    if model_type == 'attention':
        model = SimpleAttention(512, 500, remat = True).cuda()
    elif model_type == 'cnn':
        model = SimpleCNN(remat = True).cuda()

    # inputs = torch.cat([inputs.detach(), inputs.detach()], dim = 0).cuda()
    # labels = torch.cat([labels.detach(), labels.detach()], dim = 0).cuda()
    # inputs.requires_grad = True
    inputs, labels = generate_inputs(model_type, batch_size * 2)

    torch.cuda.synchronize()
    
    times = []
    for _ in range(iters):
        start_event = torch.cuda.Event(enable_timing = True)
        end_event = torch.cuda.Event(enable_timing = True)
        start_event.record()
        run((model, inputs, labels, optimizer), update = True)
        end_event.record()
        end_event.synchronize()
        time_taken = start_event.elapsed_time(end_event) / 1000  # convert to seconds
        times.append(time_taken)
    
    torch.cuda.cudart().cudaProfilerStop()
        
    median = sorted(times)[iters // 2]
    print(f'With remat, batch size = {batch_size} : {median:.2f}s.\nThroughput = {batch_size // median} img/s.')
