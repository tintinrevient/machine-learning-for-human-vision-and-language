import imageio
import torch

dir_path = '../data/volumetric-dicom'
vol_array = imageio.volread(dir_path, 'DICOM')

vol_tensor = torch.from_numpy(vol_array).float()
vol_tensor = torch.transpose(vol_tensor, 0, 2)
vol_tensor = torch.unsqueeze(vol_tensor, 0)

print(vol_tensor.shape)