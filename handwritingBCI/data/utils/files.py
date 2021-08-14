import numpy as np
from scipy.io import loadmat
from handwritingBCI.pathlib_extension import Path


def get_dataset(path):
    data, labels = get_data(path)
    data = np.stack(data)
    data, labels = get_shuffled_data_labels(data, labels)
    return data, labels


def get_data(path):
    data, labels = [], []
    for file in get_files(path):
        file_data, file_labels = get_file_data(file)
        data += file_data
        labels += file_labels
    return data, labels


def get_file_data(file):
    file_dict = read_file(file)
    file_data, file_labels = extract_relevant_data_labels(file_dict)
    return file_data, file_labels


def read_file(file):
    return loadmat(str(file))


def extract_relevant_data_labels(file_dict):
    file_data, file_labels = [], []
    for key in file_dict.keys():
        if is_correct_key(key):
            key_data, key_labels = extract_key_data(file_dict, key)
            file_data += key_data
            file_labels += key_labels
    return file_data, file_labels


def extract_key_data(file_dict, key):
    # Get the data for the particular character
    key_data = file_dict[key]

    # key_data is numpy array of shape
    # num_trials X time_steps X num_electrodes
    # split all trials

    key_data = [trial_data for trial_data in key_data]

    # extract the label for these trials from the key
    trial_label = extract_label(key)
    # assign original label to each trial
    key_labels = [trial_label] * len(key_data)
    return key_data, key_labels


def extract_label(key):
    return key.split("_")[1]


def is_correct_key(key):
    return "neuralActivityCube" in key and "doNothing" not in key


def get_shuffled_data_labels(data, labels):
    index = get_random_index(data)
    data = data[index]
    labels = [labels[i] for i in index]
    return data, labels


def get_random_index(data):
    return np.random.permutation(data.shape[0])


def get_files(path, filename="singleLetters.mat"):
    if isinstance(path, str):
        path = Path(path)
    return path.ls(recurse=True, include=[filename])
