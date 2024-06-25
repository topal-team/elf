import torch
import torch.nn as nn 

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

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

    def get_sample(self, batch_size):
        return torch.randn(batch_size, 3, 224, 224)

class SimpleAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SimpleAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

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

        return context

    def get_sample(self, batch_size):
        return torch.randn(batch_size, 64, self.hidden_dim)
    
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_blocks = 4):
        super(SimpleTransformer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, input_dim)
        self.blocks = []
        for i in range(n_blocks):
            self.blocks.append(nn.Sequential(SimpleAttention(hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                nn.Linear(hidden_dim, hidden_dim)))
            self.add_module(f'block_{i}', self.blocks[-1])
    
    def forward(self, x):
        x = self.embed(x)

        for b in self.blocks:
            x = b(x)
        x = self.head(x)
        return x.view(-1, x.size(-1))

    
    def get_sample(self, batch_size):
        return torch.randint(0, self.input_dim, (batch_size, 64, self.hidden_dim))