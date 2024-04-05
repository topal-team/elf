import torch
import torch.nn as nn


class SubModule(nn.Module):
    def __init__(self, linear):
        super(SubModule, self).__init__()
        self.embedding = linear
        self.fc = nn.Linear(200, 200)

    def forward(self, input):
        return self.fc(self.embedding(input))
    
    def change_embed_weights(self, num):
        nn.init.constant_(self.embedding.weight, num)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.embedding = nn.Linear(5, 5)
        self.net_a = SubModule(self.embedding)
        self.net_b = SubModule(self.embedding)

    def forward(self, input):
        return self.net_a(input) + self.net_b(input)
    
    def is_embed_weights_equal(self):
        return torch.all((self.net_a.embedding.weight-self.net_b.embedding.weight) == 0)

    def print_embed_weights(self):
        print(self.net_a.embedding.weight)
        print(self.net_b.embedding.weight)

m = Model()
print('named params')
for n, p in m.named_parameters():
    print(n, p.shape, p.data_ptr)

print('params')
for p in m.parameters():
    print(p.shape, p.data_ptr)

print('state dict')
for n, p in m.state_dict().items():
    print(n, p.shape, p.data_ptr)

# TO CHANGE WEIGHTS AND CHECK EQUALITY
# before saving
m.net_a.change_embed_weights(1)
m.print_embed_weights()
print(m.is_embed_weights_equal())

# save the model
torch.save(m.state_dict(), 'sample.pth')

new_m = Model()
new_m.load_state_dict(torch.load('sample.pth'))

# before saving
new_m.net_b.change_embed_weights(2)
new_m.print_embed_weights()
print(new_m.is_embed_weights_equal())
