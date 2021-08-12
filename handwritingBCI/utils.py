import numpy as np
from scipy.io import loadmat


def get_data(path, filename="singleLetters.mat"):
    return path.ls(recurse=True, include=[filename])


def get_dataset(path):
    dataset, labels = [], []
    for path in get_data(path):
        data = loadmat(str(path))
        for key in data.keys():
            if "neuralActivityCube" in key and "doNothing" not in key:
                dataset += [d for d in data[key]]
                labels += [key.split("_")[1] for _ in range(len(data[key]))]
    dataset = np.stack(dataset)
    return dataset, labels
