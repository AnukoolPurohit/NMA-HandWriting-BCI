from handwritingBCI import Path
from handwritingBCI.data.utils import get_data


def test_get_data(test_path):
    path = Path(test_path)
    assert len(get_data(path)) == 10
    return
