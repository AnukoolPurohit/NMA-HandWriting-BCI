from handwritingBCI.pathlib_extension import Path


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

