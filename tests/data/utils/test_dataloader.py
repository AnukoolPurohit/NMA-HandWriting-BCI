import pytest
from tests.macros import test_parameters
from handwritingBCI.data.utils.dataloader import (get_train_test_lengths,
                                                  get_neuro_dataloaders)


def test_get_neuro_dataloaders(test_path):
    train_neuro_dl, valid_neuro_dl = get_neuro_dataloaders(test_path)
    train_x, train_y = next(iter(train_neuro_dl))
    valid_x, valid_y = next(iter(valid_neuro_dl))

    assert train_x.shape == (64, 1, 201, 192)
    assert train_x.shape == valid_x.shape
    assert train_y.shape == valid_y.shape
    return


@pytest.mark.parametrize("data_length,test_size,expected", test_parameters)
def test_get_train_test_lengths(data_length, test_size, expected):
    assert get_train_test_lengths(data_length, test_size) == expected
    return
