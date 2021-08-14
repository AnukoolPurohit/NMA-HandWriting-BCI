from torch.utils.data import Dataset


class NeuroDataset(Dataset):
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
