from pathlib import Path
import requests

import pytest

from . import null_hash
from .asset_handler import RMAfterTest
from ..utils import loader


def test_check():
    """Write an empty file."""
    dfile = (Path(__file__).resolve().parent / Path('assets/test_file_check.txt'))

    if dfile.exists():
        raise RuntimeError("File exists. Test setup failed.")

    """Assertions"""
    with RMAfterTest(dfile):
        dfile.touch()  # write file

        assert loader.check_file(dfile, None)
        assert loader.check_file(dfile, null_hash)
        assert not loader.check_file(dfile, "a_wrong_hash")
        assert not loader.check_file(Path("a_wrong_file"), null_hash)  # but the right hash
        assert not loader.check_file(Path("a_wrong_file"), "a_wrong_hash")  # both wrong


def test_load():

    """Raise if URL not okay"""
    with pytest.raises(requests.exceptions.HTTPError):
        loader.load('idk.txt', 'https://www.embl.de//asdjfklasdjfiwuas')

    """Load fresh file"""
    fresh_file = Path(__file__).resolve().parent / Path('assets/downloadable_file.txt')
    if fresh_file.exists():
        raise ValueError("Test setup error. File should not have existed.")

    with RMAfterTest(fresh_file):
        loader.load(fresh_file, "https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ/download")
        loader.load(fresh_file, "https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ/download",
                        "ffd418a1d5bdb03a23a76421adc11543185fec1f0944b39c862a7db8e902710a")

        with pytest.raises(RuntimeError):
            loader.load(fresh_file, "https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ/download",
                     "wrong_hash")