import numpy as np
from scipy.io import loadmat


def get_dataset(path):
    total_data, labels = [], []
    for path in get_data(path):
        data = loadmat(str(path))
        for key in data.keys():
            if "neuralActivityCube" in key and "doNothing" not in key:
                total_data += [d for d in data[key]]
                labels += [key.split("_")[1] for _ in range(len(data[key]))]
    total_data = np.stack(total_data)
    total_data, labels = get_random_shuffled_data_labels(total_data, labels)
    return total_data, labels


def get_data(path, filename="singleLetters.mat"):
    return path.ls(recurse=True, include=[filename])


def get_random_shuffled_data_labels(data, labels):
    index = get_random_index(data)
    data = data[index]
    labels = [labels[i] for i in index]
    return data, labels


def get_random_index(data):
    return np.random.permutation(data.shape[0])


def get_train_test_lengths(data_length, test_size=0.1):
    test_size = int(data_length * test_size)
    train_size = data_length - test_size
    return [train_size, test_size]