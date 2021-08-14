import torch
from torch.utils.data import Dataset


class NeuroDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        assert data.shape[0] == len(target)
        self.transform = transform
        self.target_transform = target_transform
        return

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.target[index]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        if isinstance(data, torch.Tensor):
            if len(data.shape) == 2:
                data = data.unsqueeze(0)
        return data, label




