import random
import numpy as np
from handwritingBCI import Path
from handwritingBCI.data.utils.files import (get_files,
                                             get_random_index,
                                             get_shuffled_data_labels,
                                             is_correct_key,
                                             extract_label,
                                             extract_key_data,
                                             read_file,
                                             get_file_data,
                                             get_data,
                                             get_dataset)


def test_get_dataset(test_path, test_alphabet):
    data, labels = get_dataset(test_path)
    all_labels = set(labels)

    assert isinstance(data, np.ndarray)
    assert data.shape[0] == len(labels)
    assert data.shape[0] == 3100
    assert data.shape == (3100, 201, 192)

    for character in test_alphabet:
        assert character in all_labels

    assert "doNothing" not in all_labels
    return


def test_get_data(test_path, test_alphabet):
    data, labels = get_data(test_path)
    all_labels = set(labels)

    assert len(data) == len(labels)
    assert len(data) == 3100

    for character in test_alphabet:
        assert character in all_labels

    assert "doNothing" not in all_labels

    for trial_data in data:
        assert isinstance(trial_data, np.ndarray)
        assert len(trial_data.shape) == 2
    return


def test_get_file_data(test_file, test_alphabet):
    file_data, file_labels = get_file_data(test_file)
    all_labels = set(file_labels)

    assert len(file_data) == len(file_labels)
    assert len(file_data) == 310

    for character in test_alphabet:
        assert character in all_labels

    for trial_data in file_data:
        assert isinstance(trial_data, np.ndarray)
        assert len(trial_data.shape) == 2
    return


def test_read_file(test_file, test_keys):
    file_dict = read_file(test_file)

    assert isinstance(file_dict, dict)
    assert list(file_dict.keys()) == test_keys
    return


def test_extract_key_data(test_dict, test_alphabet):
    for character in test_alphabet:
        key = f"neuralActivityCube_{character}"
        key_data, key_label = extract_key_data(test_dict, key)

        assert len(key_label) == len(key_data)
        assert key_label == [character] * len(key_data)
        for trial_data in key_data:
            assert isinstance(trial_data, np.ndarray)
            assert len(trial_data.shape) == 2
    return


def test_extract_label(test_alphabet):
    character = random.choice(test_alphabet)

    assert extract_label(f"neuralActivityCube_{character}") == character
    return


def test_is_correct_key(test_alphabet):
    character = random.choice(test_alphabet)

    assert is_correct_key("Hello") is False
    assert is_correct_key(f"neuralActivityCube_{character}") is True
    return is_correct_key("neuralActivityCube_doNothing")


def test_get_shuffled_data_labels(test_data):
    data, labels = test_data
    new_data, new_labels = get_shuffled_data_labels(data, labels)

    assert data.shape == new_data.shape
    assert len(labels) == len(new_labels)
    return


def test_get_random_index(test_data):
    data, label = test_data
    random_index = get_random_index(data)

    assert data.shape[0] == len(random_index)
    assert data.shape[0] == random_index.max() + 1
    assert random_index.min() == 0
    return


def test_get_files(test_path):
    path = Path(test_path)

    assert len(get_files(path)) == 10
    return
