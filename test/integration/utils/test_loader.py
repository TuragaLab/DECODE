from pathlib import Path

import pytest
import requests

from decode import utils


def test_check(tmpdir, null_hash):
    p = Path(tmpdir) / "dummy_file.txt"
    p.touch()

    assert utils.files.check_file(p, None)
    assert utils.files.check_file(p, null_hash)
    assert not utils.files.check_file(p, "a_wrong_hash")
    assert not utils.files.check_file(Path("a_wrong_file"), null_hash)
    assert not utils.files.check_file(Path("a_wrong_file"), "a_wrong_hash")


def test_load(tmpdir):
    # raise if path not okay
    with pytest.raises(requests.exceptions.HTTPError):
        utils.files.load("idk.txt", "https://www.embl.de//asdjfklasdjfiwuas")

    # load fresh file
    fresh_file = Path(tmpdir) / "downloadable_file.txt"

    # file without hash
    utils.files.load(
        fresh_file, "https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ/download"
    )

    # file with hash
    utils.files.load(
        fresh_file,
        "https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ/download",
        "ffd418a1d5bdb03a23a76421adc11543185fec1f0944b39c862a7db8e902710a",
    )

    # assert raise on wrong hash
    with pytest.raises(RuntimeError):
        utils.files.load(
            fresh_file,
            "https://oc.embl.de/index.php/s/kK5Cx7qYlnRFnQQ/download",
            "wrong_hash",
        )
