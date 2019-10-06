import torch
import numpy
from matplotlib import pyplot as plot

def model(t_x, w, b):
    return w * t_x + b

def loss_fn(t_p, t_y):
    squared_diff = (t_p - t_y)**2
    return squared_diff.mean() # mean square loss

# d loss_fn / d w = (d loss_fn / d t_p) * (d t_p / d w)
# d loss_fn / d b = (d loss_fn / d t_p) * (d t_p / d b)

# (d loss_fn / d t_p)
def dloss_fn(t_p, t_y):
    dsquared_diff = 2 * (t_p - t_y)
    return dsquared_diff

# (d t_p / d w)
def dmodel_dw(t_x, w, b):
    return t_x

# (d t_p / d b) -> b + constant -> 1 * b**0 = 1.0
def dmodel_db(t_x, w, b):
    return 1.0

def grad_fn(t_x, t_y, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_y) * dmodel_dw(t_x, w, b)
    dloss_db = dloss_fn(t_p, t_y) * dmodel_db(t_x, w, b)
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])

def training_loop(epochs, learning_rate, params, t_x, t_y):
    for epoch in range(1, epochs + 1):
        w, b = params
        t_p = model(t_x, w, b)
        loss = loss_fn(t_p, t_y)
        grad = grad_fn(t_x, t_y, t_p, w, b)

        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        print('\t Params: ', params)
        print('\t Grad: ', grad)
    return params


t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_x = torch.tensor(t_x)
# the range of the input = [-1.0, 1.0]
t_normalized_x = 0.1 * t_x

w = torch.ones(1)
b = torch.zeros(1)

t_y = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_y = torch.tensor(t_y)

t_p = model(t_x, w, b)
loss = loss_fn(t_p, t_y)

# start training
# learning_rate = 1e-2 -> diverge
# training_loop(epochs = 100, learning_rate = 1e-2, params = torch.tensor([1.0, 0.0]), t_x = t_x, t_y = t_y)

# learning_rate = 1e-4 -> converge, but the output is unstable due to the lack of normalization for the input
# training_loop(epochs = 100, learning_rate = 1e-4, params = torch.tensor([1.0, 0.0]), t_x = t_x, t_y = t_y)

# normalization of the input
params = training_loop(epochs = 5000, learning_rate = 1e-2, params = torch.tensor([1.0, 0.0]), t_x = t_normalized_x, t_y = t_y)

# plot
t_p = model(t_normalized_x, *params) # *params = a b, params = [a, b]

figure = plot.figure(dpi=600)
plot.xlabel('Fahrenheit')
plot.ylabel('Celsius')
plot.plot(t_x.numpy(), t_p.detach().numpy())
plot.plot(t_x.numpy(), t_y.numpy(), 'o')
# plot.show()

# example is as below
delta = 0.1
learning_rate = 1e-2 # 1e-2=0.01

loss_rate_of_change_w = (loss_fn(model(t_x, w + delta, b), t_y) - loss_fn(model(t_x, w - delta, b), t_y)) / (2.0 * delta)
w = w - learning_rate * loss_rate_of_change_w

loss_rate_of_change_b = (loss_fn(model(t_x, w, b + delta), t_y) - loss_fn(model(t_x, w, b - delta), t_y) / (2.0 * delta))
b = b - learning_rate * loss_rate_of_change_b

# epochs = 10
# for epoch in range(epochs):
#     loss_rate_of_change_w = (loss_fn(model(t_x, w + delta, b), t_y) - loss_fn(model(t_x, w - delta, b), t_y)) / (2.0 * delta)
#     loss_rate_of_change_b = (loss_fn(model(t_x, w, b + delta), t_y) - loss_fn(model(t_x, w, b - delta), t_y) / (2.0 * delta))
#
#     w = w - learning_rate * loss_rate_of_change_w
#     b = b - learning_rate * loss_rate_of_change_b

