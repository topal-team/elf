import torch
import time
import numpy as np

def zz():
    x = torch.randn(10, 10, requires_grad=True)
    print(x.grad)
    l = [lambda : x.grad]
    x.sum().backward()
    print(x.grad)
    return l

l = zz()
print(l[0]())
exit(0)

def zero_grad(tensor):
    if tensor.grad is not None:
        tensor.grad.data.zero_()

x = torch.randn(2048, 4096, requires_grad=True)
w = torch.randn(4096, 4096, requires_grad=True)
z = torch.randn(2048, 4096, requires_grad=True)

n = 100

times = []
for _ in range(n):
    y = x @ w
    y2 = y.detach().requires_grad_()
    loss = (y2 - z).abs().sum()
    start = time.time()
    loss.backward()
    y.backward(y2.grad.data)
    end = time.time()
    times.append(end - start)
    zero_grad(x)
    zero_grad(w)
    zero_grad(z)

split = np.median(times)
print(f"Split backward: {split}")

times = []
for _ in range(n):
    y = x @ w
    loss = (y - z).abs().sum()
    start = time.time()
    loss.backward()
    end = time.time()
    times.append(end - start)
    zero_grad(x)
    zero_grad(w)
    zero_grad(z)

fused = np.median(times)
print(f"Fused backward: {fused}")
print(f"Speedup: {100 * (split - fused) / split:.2f}%")

