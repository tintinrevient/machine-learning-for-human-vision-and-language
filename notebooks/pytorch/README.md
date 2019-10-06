## PyTorch

### Storage

![storage](./pix/storage.png)

```
import torch

points = torch.tensor([[5, 7, 4], [1, 3, 2], [7, 3, 8]])

points.storage()
points.is_contiguous()
points.stride()
points.shape
points.dtype
points[1,].storage_offset()

points_transpose = points.t()
points_transpose = points.transpose(0, 1)

points_transpose.storage()
points_transpose.is_contiguous()
points_transpose.stride()
points_transpose.shape

id(points.storage()) == id(points_transpose.storage()) # the storage is the same, only the view is changed

points_transpose_contiguous = points_transpose.contiguous()
points_transpose_contiguous.is_contiguous()
id(points.storage()) == id(points_transpose_contiguous.storage()) # the storage is the same
points.is_contiguous()
```

### Numeric Types

The numeric types are as below:
* torch.float32 = torch.float -> torch.FloatTensor or torch.Tensor (**default**)
* torch.float64 = torch.double -> torch.DoubleTensor
* torch.float16 = torch.half
* torch.int8 -> torch.CharTensor
* torch.uint8 -> torch.ByteTensor
* torch.int16 = torch.short
* torch.int32 = torch.int
* torch.int64 = torch.long
* torch.bool -> torch.BoolTensor

The allocation of a tensor with the right numeric type:
```
import torch

double_tensor = torch.ones(10, 2, dtype=torch.double)
double_tensor = torch.ones(10, 2).double()
double_tensor = torch.ones(10, 2).to(torch.double)
double_tensor = torch.ones(10, 2).to(dtype=torch.double)

float_tensor = torch.ones(10, 2)
double_tensor = float_tensor.type(torch.double)
```

### NumPy Interoperability

The returned NumPy arrays share the same buffer with the tensor storage in CPU RAM and vice versa.
```
import torch

array_tensor = torch.ones(3, 4)
array_numpy = array_tensor.numpy()

new_array_tensor = torch.from_numpy(array_numpy)
```

### Serialization

PyTorch uses pickle to serialize the tensor object.
```
import torch

points = torch.tensor([[5, 7, 4], [1, 3, 2], [7, 3, 8]])

torch.save(points, '../data/points.t')
loaded_points = torch.load('../data/points.t')

with open('../data/points.t', 'wb') as file:
  torch.save(points, file)
  
with open('../data/points.t', 'rb') as file:
  loaded_points = torch.load(file)
```

HDF5 is a portable format to store serialized multi-dimensional arrays in a key-value dictionary.
```
import torch
import h5py

file = h5py.File('../data/points.hdf5', 'w')
file.create_dataset('key', data=points.numpy())
file.close()

file = h5py.File('../data/points.hdf5', 'r')
dataset = file['key']
last_row = dataset[-1:]
tensor_last_row = torch.from_numpy(last_row)
file.close()
```
