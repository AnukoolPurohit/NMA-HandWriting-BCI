import torch
import numpy as np
from scipy.io import loadmat
from .dataset import NeuroDataset
from .preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split


def get_neuro_dataset(path):
    data, labels = get_dataset(path)
    transform, target_transform = torch.tensor, LabelEncoder(labels)
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


def get_neuro_dataloaders(path, test_size=0.1, batch_size=64):
    neuro_dataset = get_neuro_dataset(path)
    train_ds, valid_ds = get_train_valid_split(neuro_dataset, test_size=test_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_dl, valid_dl


def get_dataset(path):
    total_data, labels = [], []
    for path in get_data(path):
        data = loadmat(str(path))
        for key in data.keys():
            if "neuralActivityCube" in key and "doNothing" not in key:
                total_data += [d for d in data[key]]
                labels += [key.split("_")[1] for _ in range(len(data[key]))]
    total_data = np.stack(total_data)
    total_data, labels = get_random_shuffled_data_labels(total_data, labels)
    return total_data, labels


def get_data(path, filename="singleLetters.mat"):
    return path.ls(recurse=True, include=[filename])


def get_random_shuffled_data_labels(data, labels):
    index = get_random_index(data)
    data = data[index]
    labels = [labels[i] for i in index]
    return data, labels


def get_random_index(data):
    return np.random.permutation(data.shape[0])


def get_train_test_lengths(data_length, test_size=0.1):
    test_size = int(data_length * test_size)
    train_size = data_length - test_size
    return [train_size, test_size]
