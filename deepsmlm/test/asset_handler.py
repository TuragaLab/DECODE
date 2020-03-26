"""Provides means to handle test assets, i.e. check if file exists and loads."""

import pathlib
from pathlib import Path

import requests


def check_load(file: (str, pathlib.Path), url: str, verbose: bool = True):
    """

    Args:
        file:
        url:
        verbose:

    Returns:
        bool: true if file already existed

    """

    if not isinstance(file, pathlib.Path):
        file = pathlib.Path(file)

    if file.exists():
        return True

    else:
        if verbose:
            print("File does not exist. Attempt to load from URL")
        file_www = requests.get(url)
        file_www.raise_for_status()  # raises an error if the file is not available
        with file.open('wb') as f:
            f.write(file_www.content)

        return False


class RMAfterTest:
    """A small helper that deletes a dummy test file after test completion."""

    def __init__(self, file):
        assert isinstance(file, Path)
        self.file = file

    def __enter__(self):
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.unlink()