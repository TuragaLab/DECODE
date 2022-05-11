import test
from pathlib import Path

import pytest

import decode.utils.files
from decode.generic import asset_handler


class TestAssetHandler:

    @pytest.fixture()
    def ass(self):
        return asset_handler.AssetHandler()

    def test_auto_load(self, tmpdir, ass):
        file = Path(tmpdir) / 'downloadable_file.txt'

        # give the file you want, and let it look it up in the yaml."""
        ass.auto_load(file)
        ass.auto_load(file)  # Repeat test with already present file

        with pytest.raises(ValueError):
            ass.auto_load(Path('non_existing.nonexist'))


def test_hash(null_hash, tmpdir):
    tmpdir = Path(tmpdir)
    dfile = tmpdir / "test_hash.txt"
    dfile.touch()

    assert null_hash == decode.utils.files.hash_file(dfile)

    with pytest.raises(FileExistsError):
        decode.utils.files.hash_file(tmpdir / Path('a_non_exist_file.nonexist'))
