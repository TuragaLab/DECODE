import pytest
from unittest import mock
from pathlib import Path

from decode.generic import asset_handler


@pytest.mark.parametrize("path", [None, "special/dir"])
def test_auto_asset(path):

    @asset_handler.auto_asset("dummy", path)
    def f(path_dummy: Path) -> Path:
        return path_dummy

    with mock.patch.object(
        asset_handler.yaml,
        "safe_load",
        return_value={
            "dummy": {"file_name": "dummy_file_name", "url": None, "hash": None}
        },
    ):
        with mock.patch.object(
            asset_handler.utils.files, "check_load", return_value=None
        ):
            p = f()

    if path is None:
        assert {asset_handler.AssetHandler._path_save_default}.issubset(set(p.parents)), \
            "Wrong default path"

    else:
        assert path in str(p)
