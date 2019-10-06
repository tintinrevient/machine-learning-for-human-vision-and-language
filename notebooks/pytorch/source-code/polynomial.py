import torch
from torch import optim

def model(t_x, w1, w2, b):
    return w2 * t_x ** 2 + w1 * t_x + b

def loss_fn(t_p, t_y):
    squared_diff = (t_p - t_y)**2
    return squared_diff.mean()

def training_loop(epochs, optimizer, params, train_t_x, val_t_x, train_t_y, val_t_y):
    for epoch in range(1, epochs + 1):
        train_t_p = model(train_t_x, *params)
        train_loss = loss_fn(train_t_p, train_t_y)

        # val_t_p = model(val_t_x, *params)
        # val_loss = loss_fn(val_t_p, val_t_y)
        with torch.no_grad():
            val_t_p = model(val_t_x, *params)
            val_loss = loss_fn(val_t_p, val_t_y)
            assert val_loss.requires_grad == False

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print('Epoch {}, Training loss {}, Validation loss {}'.format(epoch, float(train_loss), float(val_loss)))

    return params

t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_x = torch.tensor(t_x)

t_y = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_y = torch.tensor(t_y)

samples = t_x.shape[0]
vals = int(0.2 * samples)

shuffled_indices = torch.randperm(samples)

train_indices = shuffled_indices[:-vals]
val_indices = shuffled_indices[-vals:]

print(train_indices, val_indices)

train_t_x = t_x[train_indices]
train_t_y = t_y[train_indices]
val_t_x = t_x[val_indices]
val_t_y = t_y[val_indices]

train_t_normalized = 0.1 * train_t_x
val_t_normalized = 0.1 * val_t_y

params = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
learning_rate = 1e-4
optimizer = optim.SGD([params], lr=learning_rate)

params = training_loop(epochs = 5000, optimizer=optimizer, params = params,
                       train_t_x=train_t_normalized, val_t_x=val_t_normalized, train_t_y=train_t_y, val_t_y=val_t_y)