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

macro_image_sizes = [
    ((100, 100), (50, 50)),
    ((201, 196), (100, 98)),
    ((50, 50), (25, 25)),
    ((137, 299), (68, 149))
]