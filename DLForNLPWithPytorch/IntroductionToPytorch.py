import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x = torch.randn(2,2)
y = torch.randn(2,2)

z = x+y

print(z.grad_fn)

x = x.requires_grad_()
y = y.requires_grad_()

z = x+y

print(z.grad_fn)

print(z.requires_grad)

new_z = z.detach()

print(new_z.grad_fn)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)