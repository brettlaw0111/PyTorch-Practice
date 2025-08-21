import torch

#Dummy training
weights = torch.ones(4, requires_grad = True)

#PyTorch built-in optimizer. Takes a list of tensors and a learning rate(lr)
optimizer = torch.optim.SGD([weights], lr=0.01) #Stochastic Gradient Descent

#Optimization step
optimizer.step()

#Clears the gradient
optimizer.zero_grad()

#Training loop
for epoch in range(3):
	#Dummy operation that simulates model output
	model_output = (weights*3).sum()

	#Simulated model output. Calculates the gradient
	model_output.backward()

	#All values are summed up. Our weights are incorrect without emptying
	print(weights.grad)

	#Empties the gradient
	weights.grad.zero_()