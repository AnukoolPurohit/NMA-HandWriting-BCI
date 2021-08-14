import shutil
import random
import pytest
import pickle
import numpy as np
from .long_fixtures import KEYS, SAMPLE_DICT_PATH
from handwritingBCI import Path


DATA_PATH = "/home/anukoolpurohit/Documents/AnukoolPurohit/Datasets/HandwritingBCI/handwriting-bci/handwritingBCIData"
DATA_PATH = Path(DATA_PATH)
DATA_PATH = DATA_PATH/"Datasets"/"t5.2019.11.25"/"singleLetters.mat"


@pytest.fixture()
def test_path(tmp_path):
    for i in range(10):
        path = tmp_path/f"subject{i}"
        path.mkdir()
        shutil.copy(DATA_PATH, path)
        if i % 2 == 0:
            extra_path = tmp_path/f"subject{i}"/"data_to_exclude.nonsense"
            extra_path.touch()
    return str(tmp_path)


@pytest.fixture()
def test_file(test_path):
    test_file = test_path + "/subject1/singleLetters.mat"
    return test_file


@pytest.fixture()
def test_data(test_alphabet):
    number_of_data_points = random.randint(2, 12)
    labels = random.choices(test_alphabet, k=number_of_data_points)
    return np.random.randn(number_of_data_points, 201, 192), labels


@pytest.fixture()
def test_alphabet():
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    return alphabet


@pytest.fixture()
def test_dict():
    with open(SAMPLE_DICT_PATH, 'rb') as handle:
        test_dict = pickle.load(handle)
    return test_dict


@pytest.fixture()
def test_keys():
    return KEYS
