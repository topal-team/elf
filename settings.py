import torch
from models.GPT import GPTXXXLConfig as GPTConf
from models.resnet import ResNet, Bottleneck

vocab_size = 3072
placement = [0, 1, 2, 3]

dataset_size = 64
iters = 50

# inputs = torch.randint(0, vocab_size, (dataset_size, block_size))
# model = GPT(GPTConf(vocab_size, block_size))
inputs = torch.randn((dataset_size, 3, 224, 224))
model = ResNet(Bottleneck, [8] * 4)
