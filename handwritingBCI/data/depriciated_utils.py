def train_test_spilt(dataset, labels, test_size=0.1):
    """

    :param dataset:
    :param labels:
    :param test_size:
    :return:

    Depreciated
    """
    train_index, test_index = random_split_index(dataset, test_size)
    train_data, train_labels = get_data_labels_from_index(dataset, labels, train_index)
    test_data, test_labels = get_data_labels_from_index(dataset, labels, test_index)
    return (train_data, train_labels), (test_data, test_labels)


def random_split_index(data, test_size=0.1):
    """
    Depreciated
    :param data:
    :param test_size:
    :return:
    """
    index = get_random_index(data)

    test_size = int(data.shape[0] * test_size)

    train_index = index[:test_size]
    test_index = index[test_size:]

    assert len(test_index) + len(train_index) == len(index)
    return train_index, test_index


def get_data_labels_from_index(data, labels, index):
    data = data[index]
    new_labels = [labels[i] for i in index]
    return data, new_labels