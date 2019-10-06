import torch.optim as optim
import torch

# print(dir(optim))

def model(t_x, w, b):
    return w * t_x + b

def loss_fn(t_p, t_y):
    squared_diff = (t_p - t_y)**2
    return squared_diff.mean()

def training_loop(epochs, optimizer, params, t_x, t_y):
    for epoch in range(1, epochs + 1):
        t_p = model(t_x, *params)
        loss = loss_fn(t_p, t_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params

t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_x = torch.tensor(t_x)
# the range of the input = [-1.0, 1.0]
t_normalized_x = 0.1 * t_x

t_y = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_y = torch.tensor(t_y)

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

params = training_loop(epochs = 5000, optimizer=optimizer, params = params, t_x = t_normalized_x, t_y = t_y)

# example is as below
t_p = model(t_normalized_x, *params)
loss = loss_fn(t_p, t_y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(params)