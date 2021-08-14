from .files import get_dataset
from .transforms import get_transforms
from ..dataset import NeuroDataset
from torch.utils.data import DataLoader, random_split


def get_neuro_dataloaders(path, test_size=0.1, batch_size=64, generator=None):
    neuro_dataset = get_neuro_dataset(path)
    train_ds, valid_ds = get_train_valid_split(neuro_dataset, test_size, generator)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_dl, valid_dl


def get_neuro_dataset(path):
    data, labels = get_dataset(path)
    transform, target_transform = get_transforms(labels)
    neuro_dataset = NeuroDataset(data, labels,
                                 transform=transform,
                                 target_transform=target_transform)
    return neuro_dataset


def get_train_valid_split(neuro_dataset, test_size=0.1, generator=None):
    train_valid_lengths = get_train_test_lengths(len(neuro_dataset), test_size=test_size)
    if generator:
        train_ds, valid_ds = random_split(neuro_dataset, train_valid_lengths,
                                          generator=generator)
    else:
        train_ds, valid_ds = random_split(neuro_dataset, train_valid_lengths)
    return train_ds, valid_ds


def get_train_test_lengths(data_length, test_size=0.1):
    test_size = int(data_length * test_size)
    train_size = data_length - test_size
    return [train_size, test_size]
