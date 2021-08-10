"""
File monkey patches the Path from pathlib package for some extra functionality.
"""
from pathlib import Path


def filter_files(files, include=[], exclude=[], attr=None):
    if attr is None:
        operation = lambda x: str(x)
    else:
        operation = lambda x: getattr(x, attr)
    for incl in include:
        files = [file for file in files if incl in operation(file)]
    for excl in exclude:
        files = [file for file in files if excl not in operation(file)]
    return files


def ls(self, recurse=False, include=[], exclude=[], **kwargs):
    if recurse:
        files = list(self.glob("**/*"))
    else:
        files = list(self.iterdir())
    return filter_files(files, include, exclude, **kwargs)


Path.ls = ls
