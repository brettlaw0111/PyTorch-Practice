import torch

#autograd allows us to calculate gradients
#gradients are essential for model optimization

#With requires_grad set to True, PyTorch will create a computational graph for us whenever we do operations with the tensor
x = torch.randn(3, requires_grad = True)
print(x)

y = x + 2

#With the graph and a technique called backpropagation, we can calculate the gradient
#More on backpropagation later
#Graph does a forward pass to calculate y
#Since it requires a gradient, PyTorch will create and store a function for us for use in backpropagation to get the gradient
#y has the attribute grad_fm. This will point to a gradient function. (Add Backward)
#With this function, we can calculate the gradient in a "backward pass"
#dx/dy

print(y)

z = y * y * 2
print(z)

#z has the MulBackward function

z = z.mean()
print(z)

#Now z has the mean function

z.backward() # dz/dx
print(x.grad)

#By calling z.backward(), we can see the gradient in x
#If it tries to do so without a scalar, it will throw an error. 
#We can prevent PyTorch from tracking the history and calcluating the grad_fm attribute
#We can use this when updating our weights, which shouldn't be part of the gradient computation
#In a later tutorial, we will see a concrete example of the autograd package in action

#We have options to prevent tracking gradients

#Creates a new tensor that doesn't require the gradient
y = x.detach()
print(y)

#Wrap our operation to be done without the gradient
with torch.no_grad(): 
	y = x + 2
	print(y)

#Set the requires_grad value to false (remember a trailing _ will modify our variable in place)
x.requires_grad_(False)
print(x)

#Whenever we call the backward function, the gradient for the tensor will be accumulated into the .grad attribute (summed up)

