from pathlib import Path

import pytest
import requests

from .asset_handler import RMAfterTest
from ..test import asset_handler

zero_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # hash of an empty file


class TestAssetHandler:

    @pytest.fixture()
    def ass(self):
        return asset_handler.AssetHandler()

    def test_check(self, ass):

        """Write an empty file."""
        dfile = (Path(__file__).resolve().parent / Path('assets/test_file_check.txt'))

        if dfile.exists():
            raise RuntimeError("File exists. Test setup failed.")

        """Assertions"""
        with RMAfterTest(dfile):
            dfile.touch()  # write file

            assert ass.check(dfile, None)
            assert ass.check(dfile, zero_hash)
            assert not ass.check(dfile, "a_wrong_hash")
            assert not ass.check(Path("a_wrong_file"), zero_hash)  # but the right hash
            assert not ass.check(Path("a_wrong_file"), "a_wrong_hash")  # both wrong

    def test_load(self, ass):

        """Raise if URL not okay"""
        with pytest.raises(requests.exceptions.HTTPError):
            ass.load('idk.txt', 'https://www.embl.de//asdjfklasdjfiwuas')

        """Load fresh file"""
        fresh_file = Path(__file__).resolve().parent / Path('assets/downloadable_file.txt')
        if fresh_file.exists():
            raise ValueError("Test setup error. File should not have existed.")

        with RMAfterTest(fresh_file):
            assert ass.load(fresh_file, "https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ/download")
            assert ass.load(fresh_file, "https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ/download",
                            "ffd418a1d5bdb03a23a76421adc11543185fec1f0944b39c862a7db8e902710a")

            with pytest.raises(RuntimeError):
                ass.load(fresh_file, "https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ/download",
                         "wrong_hash")

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

        assert zero_hash == asset_handler.hash_file(dfile)

    with pytest.raises(FileExistsError):
        asset_handler.hash_file(cdir / Path('a_non_exist_file.nonexist'))
