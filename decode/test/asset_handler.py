"""Provides means to handle test assets, i.e. check if file exists and loads."""

import hashlib
import pathlib
import shutil

import click
import yaml

from ..utils import loader


class RMAfterTest:
    """
    A small helper that provides a context manager for test binaries, i.e. deletes them after leaving the
    with statement.
    """

    def __init__(self, path, recursive: bool = False):
        """

        Args:
            path: path to file that should be deleted after context
            recursive: if true and path is a dir, then the whole dir is deleted. Careful!
        """
        assert isinstance(path, pathlib.Path)
        self.path = path
        self.recursive = recursive

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path.exists():
            if self.path.is_file():
                self.path.unlink()
            elif self.recursive and self.path.is_dir():
                shutil.rmtree(self.path)


class AssetHandler:
    asset_list_path = pathlib.Path(__file__).resolve().parent / pathlib.Path("assets/asset_list.yaml")

    def __init__(self):

        with self.asset_list_path.open() as f:
            self.dict = yaml.safe_load(f)

        self._names = [d['name'] for d in self.dict]

    def auto_load(self, file: (str, pathlib.Path)):
        if not isinstance(file, pathlib.Path):
            file = pathlib.Path(file)

        """Check that file is in list and get index"""
        ix = self._names.index(str(file.name))

        loader.check_load(file, url=self.dict[ix]['url'], hash=self.dict[ix]['hash'])


def hash_file(file: pathlib.Path):
    """
    Hahses a file. Reads everything in byte mode even if it is a text file.

    Args:
        file (pathlib.Path): full path to file

    Returns:
        str: hexdigest sha256 hash

    """

    if not file.exists():
        raise FileExistsError(f"File {str(file)} does not exist.")

    return hashlib.sha256(file.read_bytes()).hexdigest()


@click.command()
@click.option("--file", required=True, help="Specify a file that should be hased")
def hash_file_cmd(file: str):
    """
    Wrapper function to make this script callable from command line to hash new files.
    It will print the result to the console. Will treat all files in byte mode.

    Args:
        file (str): path of file

    """
    print(hash_file(pathlib.Path(file)))


if __name__ == '__main__':
    hash_file_cmd()
