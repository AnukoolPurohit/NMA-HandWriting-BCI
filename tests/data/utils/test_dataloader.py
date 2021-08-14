import pytest
from handwritingBCI.data.utils.dataloader import (get_train_test_lengths,)
from tests.macros import test_parameters


@pytest.mark.parametrize("data_length,test_size,expected", test_parameters)
def test_get_train_test_lengths(data_length, test_size, expected):
    assert get_train_test_lengths(data_length, test_size) == expected
    return
