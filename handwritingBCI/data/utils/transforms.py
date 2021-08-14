import torch
from ..preprocessing import LabelEncoder


def to_float_tensor(data):
    data = torch.tensor(data)
    data = add_channel_dimension(data)
    return data


def add_channel_dimension(data):
    if len(data.shape) == 2:
        data = data.unsqueeze(0)
    return data


def get_transforms(labels):
    return to_float_tensor, LabelEncoder(labels)
