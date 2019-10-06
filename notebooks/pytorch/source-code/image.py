import imageio
import os
import torch

image_array = imageio.imread('../data/bobby.jpg')
image_tensor = torch.from_numpy(image_array)
image_tensor = torch.transpose(image_tensor, 0, 2)

batch_size = 100
batch = torch.zeros(100, 3, 256, 256, dtype=torch.uint8)

data_dir = '../data/image-cats/'
filenames = [filename for filename in os.listdir(data_dir) if os.path.splitext(filename)[-1] == '.png']
for i, filename in enumerate(filenames):
    image_array = imageio.imread(filename)
    batch[i] = torch.transpose(torch.from_numpy(image_array), 0, 2)

batch = batch.float()
batch /= 255.0

channels = batch.shape[1]
for channel in range(channels):
    mean = torch.mean(batch[:, channel])
    std = torch.std(batch[:, channel])
    batch[:, channel] = (batch[:, channel] - mean) / std