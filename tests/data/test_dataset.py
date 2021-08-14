import random
import pytest
from tests.macros import test_transforms
from handwritingBCI.data.dataset import NeuroDataset


class TestNeuroDataset:
    def test_len(self, test_data):
        data, _ = test_data
        neuro_dataset = self.get_neuro_dataset(test_data)

        assert len(neuro_dataset) == data.shape[0]
        return

    def test_get_item(self, test_data):
        data, label = test_data
        neuro_dataset = self.get_neuro_dataset(test_data)

        trial = random.randint(0, len(neuro_dataset)-1)
        trial_data, trial_label = neuro_dataset[trial]

        assert trial_label == label[trial]
        assert trial_data.shape == data[trial].shape
        return

    @pytest.mark.parametrize("transform,target_transform", test_transforms)
    def test_apply_transform(self, test_data, transform, target_transform):
        data, label = test_data
        neuro_dataset = self.get_neuro_dataset(test_data,
                                               transform, target_transform)

        trial = random.randint(0, len(neuro_dataset)-1)
        trial_data, trial_label = neuro_dataset[trial]

        assert trial_data == transform(data[trial])
        assert trial_label == target_transform(label[trial])
        return

    def test_from_path(self, test_path):
        neuro_dataset = NeuroDataset.from_path(test_path)
        trial = random.randint(0, len(neuro_dataset)-1)
        trial_data, trial_label = neuro_dataset[trial]

        assert len(neuro_dataset) == 3100
        assert trial_data.shape == (1, 201, 192)
        assert isinstance(trial_label, int)
        return

    @staticmethod
    def get_neuro_dataset(test_data, transform=None, target_transform=None):
        data, labels = test_data
        neuro_dataset = NeuroDataset(data, labels, transform, target_transform)
        return neuro_dataset
