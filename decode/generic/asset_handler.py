"""Provides means to handle test assets, i.e. check if file exists and loads."""

from pathlib import Path
from typing import Optional, Union
from functools import wraps

import click
import yaml

from decode import utils


class AssetHandler:
    _path_repo = Path(__file__).resolve().parents[2]
    _path_asset_list = _path_repo / "test/assets/asset_list.yaml"
    _path_save_default = _path_repo / "test/assets"

    def __init__(self, assets: Optional[dict] = None):

        if assets is not None:
            self._assets = assets
        else:
            with self._path_asset_list.open() as f:
                self._assets = yaml.safe_load(f)

    def auto_load(self, name: str, path: Path) -> Path:
        asset = self._assets[name]

        if path is None:
            path = self._path_save_default / asset["file_name"]

        utils.files.check_load(path, url=asset["url"], hash=asset["hash"])
        return path


def load_asset(name, path: Optional[Path] = None) -> Path:
    """
    Load an asset by name.

    Args:
        name: name of the asset
        path: save path

    Returns:
        path to which asset was saved.
    """
    return AssetHandler().auto_load(name=name, path=path)


def auto_asset(name, path: Optional[Path] = None, return_path: bool = True):
    """
    Decorator to automatically retrieve asset and insert path into decorated functions
    signature (default).

    Args:
        name:
        path:
        return_path: insert return path as "path_{name}" in decorated function's signature

    Examples:

        @auto_asset("bead_cal")
        def some_fn(path_bead_cal: Path):
            pass
    """
    def wrap_fn(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            p = load_asset(name=name, path=path)
            if return_path:
                # add asset path to decorated functions signature
                kwargs.update({f"path_{name}": p})
            return fn(*args, **kwargs)
        return wrapped_fn
    return wrap_fn


@click.command()
@click.option("--file", required=True, help="Specify a file that should be hased")
def hash_file_cmd(file: str):
    """
    Wrapper function to make this script callable from command line to hash new files.
    It will print the result to the console. Will treat all files in byte mode.

    Args:
        file (str): path of file

    """
    print(utils.files.hash_file(Path(file)))


if __name__ == "__main__":
    hash_file_cmd()
