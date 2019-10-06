import torch.nn as nn
from torch import optim
import torch

def training_loop(epochs, optimizer, model, loss_fn, train_t_x, val_t_x, train_t_y, val_t_y):
    for epoch in range(1, epochs + 1):
        t_p_train = model(train_t_x)
        loss_train = loss_fn(t_p_train, train_t_y)

        t_p_val = model(val_t_x)
        loss_val = loss_fn(t_p_val, val_t_y)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print('Epoch {}, Training loss {}, Validation loss {}'.format(epoch, float(loss_train), float(loss_val)))

linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

t_x = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_x = torch.tensor(t_x).unsqueeze(1) * 0.1

t_y = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_y = torch.tensor(t_y).unsqueeze(1)

samples = t_x.shape[0]
vals = int(0.2 * samples)

shuffled_indices = torch.randperm(samples)

train_indices = shuffled_indices[:-vals]
val_indices = shuffled_indices[-vals:]

train_t_x = t_x[train_indices]
train_t_y = t_y[train_indices]
val_t_x = t_x[val_indices]
val_t_y = t_y[val_indices]

training_loop(epochs = 3000, optimizer = optimizer, model = linear_model, loss_fn = nn.MSELoss(),
    train_t_x = train_t_x, val_t_x = val_t_x, train_t_y = train_t_y, val_t_y = val_t_y)

print()
print(linear_model.weight)
print(linear_model.bias)

# example is as below

# linear_model = nn.Linear(1, 1) # 1 input feature & 1 output feature
#
# x = torch.ones(10, 1)
# output = linear_model(x)
#
# print(output)
# print(linear_model.weight)
# print(linear_model.bias)