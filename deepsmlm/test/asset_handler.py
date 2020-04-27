"""Provides means to handle test assets, i.e. check if file exists and loads."""

import pathlib
import hashlib
import requests
import click
import yaml


class RMAfterTest:
    """
    A small helper that provides a context manager for test binaries, i.e. deletes them after leaving the
    with statement.
    """

    def __init__(self, file):
        assert isinstance(file, pathlib.Path)
        self.file = file

    def __enter__(self):
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file.exists():
            self.file.unlink()


class AssetHandler:

    asset_list_path = pathlib.Path(__file__).resolve().parent / pathlib.Path("assets/asset_list.yaml")

    def __init__(self):

        with self.asset_list_path.open() as f:
            self.dict = yaml.safe_load(f)

        self._names = [d['name'] for d in self.dict]

    @staticmethod
    def check(file: (str, pathlib.Path), hash=None):
        """
        Checks if a file exists and if the sha256 hash is correct

        Args:
            file:
            hash:

        Returns:
            bool:   true if file exists and hash is correct (if specified), false otherwise

        """

        if not isinstance(file, pathlib.Path):
            file = pathlib.Path(file)

        if file.exists():

            if hash is None:
                return True
            else:
                if hash == hashlib.sha256(file.read_bytes()).hexdigest():
                    return True
                else:
                    return False

        else:
            return False

    @staticmethod
    def load(file: (str, pathlib.Path), url: str, hash: str = None):

        if not isinstance(file, pathlib.Path):
            file = pathlib.Path(file)

        file_www = requests.get(url)
        file_www.raise_for_status()  # raises an error if the file is not available

        with file.open('wb') as f:
            f.write(file_www.content)

        if hash is not None and not hash == hashlib.sha256(file.read_bytes()).hexdigest():
            raise RuntimeError("Downloaded file does not match hash.")

        return True

    def check_load(self, file: (str, pathlib.Path), url: str, hash: str = None):

        if not self.check(file, hash):
            self.load(file, url, hash)

    def auto_load(self, file: (str, pathlib.Path)):

        if not isinstance(file, pathlib.Path):
            file = pathlib.Path(file)

        """Check that file is in list and get index"""
        ix = self._names.index(str(file.name))

        self.check_load(file, url=self.dict[ix]['url'], hash=self.dict[ix]['hash'])


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
