import torch
from torchvision.transforms import Compose, ToTensor, ToPILImage
from ..preprocessing import LabelEncoder


def get_cnn_transforms(labels):
    cnn_transforms = Compose([ToPILImage(), ToTensor()])
    return cnn_transforms, LabelEncoder(labels)


def get_simple_transforms(labels):
    return torch.tensor, LabelEncoder(labels)
