import torch
from ..preprocessing import LabelEncoder


def get_transforms(labels):
    return torch.tensor, LabelEncoder(labels)
