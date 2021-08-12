import pytest
from handwritingBCI.pathlib_extension import Path


@pytest.fixture()
def test_path(tmp_path):
    for i in range(10):
        path = tmp_path/f"subject{i}"/"data.mat"
        path.parent.mkdir()
        path.touch()
        if i % 2 == 0:
            extra_path = tmp_path/f"subject{i}"/"data_to_exclude.nonsense"
            extra_path.touch()
    return str(tmp_path)


class TestPath:
    def test_ls(self, test_path):
        path = Path(test_path)
        assert len(path.ls()) == 10
        return

    def test_ls_recurse(self, test_path):
        path = Path(test_path)
        assert len(path.ls(recurse=True)) == 25
        return

    def test_ls_recurse_exclude(self, test_path):
        path = Path(test_path)
        assert len(path.ls(recurse=True, exclude=[".nonsense"])) == 20
        return

    def test_ls_recurse_include(self, test_path):
        path = Path(test_path)
        assert len(path.ls(recurse=True, include=[".mat"])) == 10
        return

    def test_ls_recurse_include_output(self, test_path):
        path = Path(test_path)
        correct = 0
        for p in path.ls(recurse=True, include=[".mat"]):
            if p.name.split(".")[1] == "mat":
                correct += 1
        assert correct == 10
        return
