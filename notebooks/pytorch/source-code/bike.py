import numpy
import torch

bikes_numpy = numpy.loadtxt("../data/hour-fixed.csv",
                            dtype=numpy.float32, delimiter=",", skiprows=1, converters={1: lambda x: float(x[8:10])})

bikes_tensor = torch.from_numpy(bikes_numpy)

first_day = bikes_tensor[:24].long()
weather_one_hot = torch.zeros(first_day.shape[0], 4)
weather_one_hot.scatter_(dim=1, index=first_day[:,9].unsqueeze(1) - 1, value=1.0)
torch.cat((bikes_tensor[:24], weather_one_hot), 1)[:1]

daily_bikes = bikes_tensor.view(-1, 24, bikes_tensor.shape[1]) # -1:placeholder, 24:rows, 17:columns
daily_bikes = daily_bikes.transpose(1, 2)
daily_weather_one_hot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
daily_weather_one_hot.scatter_(dim=1, index=daily_bikes[:,9,:].long().unsqueeze(1) - 1, value=1.0)
daily_bikes = torch.cat((daily_bikes, daily_weather_one_hot), dim=1)
daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0