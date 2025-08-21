import torch
import numpy as np
#Create tensors
#convert from numport arrays to tensors & vice versa
#1d, 2d, 3d and more

#Amount of variables determines the dimensions of the tensor
x = torch.empty(2,2,3)
print(x)

#rand can be replaced with torch.zeros or torch.ones
#dtype can be replaced with different types for numbers
y = torch.rand(2,2, dtype=torch.float16)
print(y)
print(y.dtype)
print(y.size())

x = torch.rand(2,2)
y = torch.rand(2,2)
#Adds the values of both tensors. Can also be accomplished with torch.add(x,y) or y.add_(x)
#Every function with a trailing underscore is an in-place operation, which modifies the current variable
z = x + y
print(x)
print(y)
print(z)
#Subtraction, multiplication and division are also possible

#Tensors support slicing operations
a = torch.rand(5,3)
print(a)
#All rows, but only the first column
print(a[:, 0])
#Row 1 with all columns
print(a[1, :])
#Element position 1,1. item() method returns  value, but can only be used with a tensor with one element
print(a[1,1].item())

#Reshaping function. Number of elements must be the same (5*3=15)
b = a.view(15)
print(b)

print(b.size())

#Can convert array to numpy array
a = torch.ones(5)
b = a.numpy()
print(b)
print(type(b))
#Gotta be careful.
#If the tensor is on the CPU and not the GPU, both objects will share the same memory location.
#If you modify, a, b will change too, and vice versa.
a.add_(1)
print(b)

#Convert numpy array to tensor
#These have the same quirk above where they share tge same memory location
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    #Creates a tensor on the GPU
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    #If you call numpy, it will give an error since it only works with CPU
    #You would have to move the tensor back to the CPU
    z = x + y
    z.to("cpu")
    print("Did the thing")

#Requires Gradient
#Tells pyTorch it needs to calculate the gradients of the tensor later in optimization steps
#If you have a variable you wanna optimize, you need the gradients
x = torch.ones(5, requires_grad=True)
print(x)