from __future__ import print_function
import torch


#Construct a 5x3 matrix
x = torch.empty(5,3)
#print(x)

#Construct a randomly initialized matrix
x = torch.rand(5,3)
#print(x)

x = torch.tensor([5.5, 3])
#print(x)

x = x.new_ones(5, 3, dtype=torch.double)
#print(x)

x = torch.randn_like(x, dtype=torch.float)
#print(x)

y = torch.rand(5, 3)

# 3 ways to add 2 tensors of same size
#1
#print(x + y)

#2
#print(torch.add(x, y))

#3
# result = torch.empty(5, 3)
# torch.add(x, y, out=result)
# print(result)

#In-place addition
y.add_(x)
#print(y)

#print(x)
#print(x[:,1])




x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
#print(x.size(), y.size(), z.size())
#print(x,y,z)


x = torch.randn(1)
#print(x)
#Get value of one element tensor as a Python number - .item()
#print(x.item())

#Tensor <-> NumPy
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
# a.add_(1)
# print(a)
# print(b)


import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)