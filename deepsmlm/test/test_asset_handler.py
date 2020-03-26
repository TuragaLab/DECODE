from pathlib import Path

import pytest
import requests

from .asset_handler import RMAfterTest
from ..test import asset_handler


def test_check_load():
    """Setup"""
    cdir = Path(__file__).resolve().parent

    """File exists, do nothing"""
    dfile = (cdir / Path('assets/test_asset_handler_dummy.txt'))
    dfile.touch()  # write file

    with RMAfterTest(dfile):
        assert asset_handler.check_load(dfile, 'https://google.com')

    """File does not exist, download possible"""
    ne_file = cdir / Path('assets/downloadable_file.txt')
    if ne_file.exists():
        raise ValueError("Test setup error. File should not have existed.")

    out = asset_handler.check_load(ne_file, 'https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ')
    assert not out, "File was present beforehand ..."
    ne_file.unlink()

    """File does not exists, URL does not exists"""
    nnee_file = cdir / Path('assets/non_existing_file.txt')

    with pytest.raises(requests.exceptions.HTTPError):
        asset_handler.check_load(nnee_file, 'https://www.embl.de//asdjfklasdjfiwuas')
