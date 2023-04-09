import torch

# x = torch.randn(3, requires_grad=True)
# print(x)

# y = x+2
# print(y)

# z = y*y*2
# # z = z.mean()
# print(z)

# # z.backward()  # dz/dx
# # print(x.grad)  # grad can be implicitly created only for scaler outputs

# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# z.backward(v)  # dz/dx
# print(x.grad)

# training examples
weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()  # need to zero out the gradient after each epoch

# example ussing optimizer
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()  # update the weights
optimizer.zero_grad()  # zero out the gradient

# when calculating gradients must specify requires grad = True then can calculate gradients with backward function
# before doing next operation in epoch must call zero function again to empty gradient
