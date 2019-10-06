import csv
import numpy
import torch

wine_path = '../data/winequality-white.csv'

wine_numpy = numpy.loadtxt(wine_path, dtype=numpy.float32, delimiter=";", skiprows=1)
attr_list = next(csv.reader(open(wine_path), delimiter=";"))

# print(wine_numpy.shape, attr_list)

wine_tensor = torch.from_numpy(wine_numpy)

# print(wine_tensor.shape, wine_tensor.type())

data = wine_tensor[:, :-1]
target = wine_tensor[:, -1].long()
target_one_hot = torch.zeros(target.shape[0], 10)
target_one_hot.scatter_(1, target.unsqueeze(1), 1.0) # dim=1, unsqueeze(1)=columns, 1.0=value

# print(target_one_hot)
# print(target.unsqueeze(1)[0,0])

data_mean = torch.mean(data, dim=0) # mean for each column
data_var = torch.var(data, dim=0)
data_normalized = (data - data_mean) / torch.sqrt(data_var)

bad_indexes = torch.le(target, 3)
# bad_indexes = (target <= 3)
# print(bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum())

bad_data = data[bad_indexes]
# print(bad_data.shape)

bad_data = data[torch.le(target, 3)]
mid_data = data[torch.gt(target, 3) & torch.lt(target, 7)]
good_data = data[torch.ge(target, 7)]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(attr_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
actual_indexes = torch.gt(target, 5)

n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
print(n_matches, n_matches / n_predicted, n_matches / n_actual)