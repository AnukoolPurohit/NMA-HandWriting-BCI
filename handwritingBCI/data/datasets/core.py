import torch
from handwritingBCI.data.utils.dataloader import get_train_test_lengths
from .base import BaseDataset
from handwritingBCI.data.utils.files import get_dataset
from handwritingBCI.data.utils.transforms import get_cnn_transforms
from torch.utils.data import DataLoader, random_split


class NeuroDataset(BaseDataset):
    @classmethod
    def from_path(cls, path, get_transforms=get_cnn_transforms):
        data, labels = get_dataset(path)
        transforms, target_transform = get_transforms(labels)

        return cls(data, labels, transforms, target_transform)

    def get_dataloaders(self, test_size=0.1, batch_size=64, generator=None):
        train_ds, valid_ds = self.get_train_valid_split(test_size, generator)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        return train_dl, valid_dl

    def get_train_valid_split(self, test_size=0.1, generator=None):
        train_valid_lengths = get_train_test_lengths(len(self),
                                                     test_size=test_size)
        if generator is None:
            generator = torch.default_generator

        train_ds, valid_ds = random_split(self, train_valid_lengths,
                                          generator=generator)
        return train_ds, valid_ds


class SeqRNNDataset(NeuroDataset):
    def __getitem__(self, index):
        data = self.data[index]
        data1 = data[:, :96]
        data2 = data[:, 96:]
        if self.transform:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
        return data1, data2
