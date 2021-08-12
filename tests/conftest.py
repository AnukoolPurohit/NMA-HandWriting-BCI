import shutil
import pytest
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
