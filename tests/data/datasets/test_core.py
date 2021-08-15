import random
from handwritingBCI.data.datasets.core import NeuroDataset


class TestNeuroDataset:
    def test_from_path(self, test_path):
        neuro_dataset = NeuroDataset.from_path(test_path)
        trial = random.randint(0, len(neuro_dataset)-1)
        trial_data, trial_label = neuro_dataset[trial]

        assert len(neuro_dataset) == 3100
        assert trial_data.shape == (1, 201, 192)
        assert isinstance(trial_label, int)
        return

    def test_get_neuro_dataloaders(self, test_path):
        neuro_dataset = NeuroDataset.from_path(test_path)
        train_neuro_dl, valid_neuro_dl = neuro_dataset.get_dataloaders()

        train_x, train_y = next(iter(train_neuro_dl))
        valid_x, valid_y = next(iter(valid_neuro_dl))

        assert train_x.shape == (64, 1, 201, 192)
        assert train_x.shape == valid_x.shape
        assert train_y.shape == valid_y.shape
        return