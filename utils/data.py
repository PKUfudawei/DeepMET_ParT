import torch
import awkward as ak
import numpy as np
import torch.nn.functional as F

from torch.utils import data
from einops import rearrange

def generalized_image_to_b_xy_c(tensor):
    """
    Transpose the tensor from [batch, channels, ..., pixel_x, pixel_y] to [batch, pixel_x*pixel_y, channels, ...]. We assume two pixel dimensions.
    """
    num_dims = len(tensor.shape) - 3  # Subtracting batch and pixel dimensions
    pattern = 'b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y -> b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)])
    return rearrange(tensor, pattern)


def generalized_b_xy_c_to_image(tensor, pixels_x=None, pixels_y=None):
    """
    Transpose the tensor from [batch, pixel_x*pixel_y, channels, ...] to [batch, channels, ..., pixel_x, pixel_y] using einops.
    """
    if pixels_x is None or pixels_y is None:
        pixels_x = pixels_y = int(np.sqrt(tensor.shape[1]))
    num_dims = len(tensor.shape) - 2  # Subtracting batch and pixel dimensions (NOTE that we assume two pixel dimensions that are FLATTENED into one dimension)
    pattern = 'b (x y) ' + ' '.join([f'c{i}' for i in range(num_dims)]) + f' -> b ' + ' '.join([f'c{i}' for i in range(num_dims)]) + ' x y'
    return rearrange(tensor, pattern, x=pixels_x, y=pixels_y)


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


def pad(x, maxlen, value=0):
    if type(x) not in [torch.Tensor, np.ndarray, ak.Array]:
        raise TypeError(f"Unsupported type: {type(x)}. Supported types are torch.Tensor, np.ndarray, and ak.Array.")
    
    if x.ndim == 1:
        x = x[:, None]

    if isinstance(x, torch.Tensor):
        x = F.pad(x, (0, maxlen - x.shape[1]), mode='constant', value=value)
    elif isinstance(x, np.ndarray):
        x = np.pad(x, (0, maxlen - x.shape[1]), mode='constant', constant_values=value)
    elif isinstance(x, ak.Array):
        x = ak.fill_none(ak.pad_none(x, target=maxlen, axis=-1, clip=True), value)
    
    return x


class Dataset(data.Dataset):
    def __init__(self, files, features, targets, start_ratio=0, end_ratio=0.001, maxlen=256, pad_value=0):
        super().__init__()

        # load data
        self.features, self.targets = [], []
        for f in files:
            if f.endswith('.parquet'):
                file_data = ak.from_parquet(f, columns=features+targets)
                file_size = len(file_data)
                start, end = int(file_size*start_ratio), int(file_size*end_ratio)
                file_data = file_data[start:end]
                file_features = [file_data[k] for k in features]
                file_targets = [file_data[k] for k in targets]
                self.features.append(file_features)
                self.targets.append(file_targets)
                print(f'Loaded {file_size} datapoints from {f}')

        self.features = ak.concatenate(self.features, axis=1)
        self.features = pad(self.features, maxlen=maxlen, value=pad_value).to_numpy().transpose(1, 0, 2)
        self.targets = ak.concatenate(self.targets, axis=1).to_numpy().T
        #as_lists = ak.to_list(self.data)
        #out = np.full((len(as_lists), maxlen), pad_value, dtype=np.float32)
        #for i, row in enumerate(as_lists):
        #    n = min(len(row), 256)
        #    out[i, :n] = row[:n]
        
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.num_datapoints = len(self.features)


    def __len__(self):
        return self.num_datapoints


    def __getitem__(self, index):
        if index >= self.num_datapoints:
            raise IndexError('index out of range')
        return self.features[index], self.targets[index]


    def transform(self, target_mean=None, target_std=None):
        self.channel_mean = {}
        self.channel_std = {}
        transformed_data = self.data.clone()
        for c in range(self.data.shape[1]):
            self.channel_std[c] = torch.std(self.data[:, c]).item() / (target_std[c] if target_std is not None else 1)
            self.channel_mean[c] = torch.mean(self.data[:, c]).item() - (target_mean[c] if target_mean is not None else 0) * self.channel_std[c]
            transformed_data[:, c] = (self.data[:, c] - self.channel_mean[c]) / self.channel_std[c]
        
        self.data = transformed_data
        return transformed_data
    
    
    def inverse_transform(self, transformed_data=None):
        if transformed_data is None:
            transformed_data = self.data
        assert transformed_data.ndim == 4
        original_data = transformed_data.clone()
        for c in range(transformed_data.shape[1]):
            original_data[:, c] = transformed_data * self.channel_std[c] + self.channel_mean[c]
        
        self.data = original_data
        return original_data