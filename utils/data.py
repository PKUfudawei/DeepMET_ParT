import torch, h5py
import awkward as ak
import numpy as np
import torch.nn.functional as F

from torch.utils import data


def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data


class Dataset(data.Dataset):
    def __init__(self, files, max_PF_num=None, dtype=torch.float32):
        super().__init__()
        self.files = [files] if type(files) is not list else files
        self.max_PF_num = max_PF_num
        self.dtype = dtype

        # lazy loading
        self.lazy_files = [h5py.File(f, 'r') for f in self.files]
        self.features = self.lazy_files[0].attrs['PF_features'].decode('utf-8').split(',')
        self.truths = self.lazy_files[0].attrs['event_truths'].split(',')

        self.file_sizes = [f.attrs['n_events'] for f in self.lazy_files]
        self.accumulated_sizes = np.cumsum([0] + self.file_sizes)
        self.total_size = self.accumulated_sizes[-1]
        print(f"=> Lazy loading {self.total_size} events from {len(self.files)} files")


    def __len__(self):
        return self.total_size


    def __getitem__(self, index):
        if index >= self.total_size:
            raise IndexError('index out of range')
        file_index = np.searchsorted(self.accumulated_sizes, index, side='right') - 1
        local_index = index - self.accumulated_sizes[file_index]

        located_file = self.lazy_files[file_index]
        PF_features = torch.from_numpy(located_file['PF_features'][local_index]).to(dtype=self.dtype)
        if self.max_PF_num is not None:
            PF_features = PF_features[:, :self.max_PF_num]
        event_truths = torch.from_numpy(located_file['event_truths'][local_index]).to(dtype=self.dtype)

        return PF_features, event_truths


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        

    def close(self):
        for f in self.lazy_files:
            f.close()