from torch.utils.data import Dataset
from handwritingBCI.data.utils.files import get_dataset
from handwritingBCI.data.utils.transforms import get_cnn_transforms


class BaseDataset(Dataset):
    def __init__(self, data, target,
                 transform=None,
                 target_transform=None):

        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

        assert data.shape[0] == len(target)
        return

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.target[index]
        data, label = self.apply_transforms(data, label)
        return data, label

    def apply_transforms(self, data, label):
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

    @classmethod
    def from_path(cls, path, get_transforms=get_cnn_transforms):
        data, labels = get_dataset(path)
        transforms, target_transform = get_transforms(labels)
        return cls(data, labels, transforms, target_transform)
