import numpy as np
test_parameters = [
    (10, 0.1, [9, 1]),
    (100, 0.1, [90, 10]),
    (100, 0, [100, 0]),
    (100, 1, [0, 100])
]

test_transforms = [
    (len, str.capitalize),
    (np.max, str.isalpha),
    (type, type)
]