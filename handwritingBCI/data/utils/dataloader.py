def get_train_test_lengths(data_length, test_size=0.1):
    test_size = int(data_length * test_size)
    train_size = data_length - test_size
    return [train_size, test_size]
