"""
	torch.Tensor is the central class of the package. 
	If you set its attribute .requires_grad as True, it starts to track all operations on it.
	When you finish your computation you can call .backward() and have all the gradients computed automatically. 
	The gradient for this tensor will be accumulated into .grad attribute.

	Stop a tensor from tracking - use .detach()

	Prevent tracking history - Wrap code in block of with torch.no_grad()
	Each tensor has a .grad_fn attribute that references a Function that has created the Tensor


"""

import torch

x = torch.ones(2,2, requires_grad=True)
#print(x)

y = x+2
#print(y)

z = y * y * 3
out = z.mean()

#print(z)
#print(out)

# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)


out.backward()
print(x.grad)


x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

# gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
# y.backward(gradients)

# print(x.grad)


# print(x.requires_grad)
# print((x ** 2).requires_grad)

# with torch.no_grad():
#     print((x ** 2).requires_grad)

