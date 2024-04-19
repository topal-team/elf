import torch
from models.GPT import GPT, GPTXXLConfig as GPTConf
from models.resnet import ResNet, Bottleneck

vocab_size = 3072
placement = [0, 1, 2, 3, 0, 1, 2, 3]
schedule = "afab"

batch_size = 64
split_sizes = [1, 2, 4, 8, 16, 32]
block_size = 64
iters = 15

inputs = torch.randint(0, vocab_size, (batch_size, block_size))
model = GPT(GPTConf(vocab_size, block_size))
# inputs = torch.randn((batch_size, 3, 224, 224))
# model = ResNet(Bottleneck, [16] * 4)
