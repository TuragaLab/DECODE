from pathlib import Path

import pytest

from .asset_handler import RMAfterTest
from ..test import asset_handler
from .. import test


class TestAssetHandler:

    @pytest.fixture()
    def ass(self):
        return asset_handler.AssetHandler()

    def test_auto_load(self, ass):

        """Make sure the file was not present before starting the actual tests."""
        file_wanted = Path(__file__).resolve().parent / Path('assets/downloadable_file.txt')
        if file_wanted.exists():
            raise RuntimeError('Test setup error. This file should not have existed.')

        """Give the file you want, and let it look it up in the yaml."""
        with RMAfterTest(file_wanted):
            ass.auto_load(file_wanted)
            ass.auto_load(file_wanted)  # Repeat test with already present file

        with pytest.raises(ValueError):
            ass.auto_load(Path('non_existing.nonexist'))


def test_hash():

    """Setup"""
    cdir = Path(__file__).resolve().parent

    """Write an empty file."""
    dfile = (cdir / Path('assets/test_hash.txt'))

    if dfile.exists():
        raise RuntimeError("File exists. Test setup failed.")

    with RMAfterTest(dfile):
        dfile.touch()  # write file

        assert test.null_hash == asset_handler.hash_file(dfile)

    with pytest.raises(FileExistsError):
        asset_handler.hash_file(cdir / Path('a_non_exist_file.nonexist'))
