import hashlib
import pathlib
import requests
import shutil
from pathlib import Path


def check_file(path: (str, pathlib.Path), hash=None) -> bool:

    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    if path.exists():

        if hash is None:
            return True
        else:
            if hash == hashlib.sha256(path.read_bytes()).hexdigest():
                return True
            else:
                return False

    else:
        return False


def load(path: (str, pathlib.Path), url: str, hash: str = None) -> None:
    """
    Loads file from URL (and checks hash if present)

    Args:
        path: path where to store downloaded file
        url:
        hash:

    """

    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    file_www = requests.get(url)
    file_www.raise_for_status()  # raises an error if the file is not available

    with path.open("wb") as f:
        f.write(file_www.content)

    d_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if hash is not None and not hash == d_hash:
        raise RuntimeError(
            f"Downloaded file does not match hash.\nSHA-256 of ref.: {hash}\nSHA-256 of downloaded: {d_hash}"
        )


def check_load(path: (str, pathlib.Path), url: str, hash: str = None):
    """
    Loads file freshly when check fails

    Args:
        path:
        url:
        hash:

    """

    if not check_file(path, hash):
        load(path, url, hash)


def hash_file(path: pathlib.Path) -> str:
    """
    Hash a file. Reads everything in byte mode even if it is a text file.

    Args:
        path (pathlib.Path): full path to file

    Returns:
        str: hexdigest sha256 hash

    """

    if not path.exists():
        raise FileExistsError(f"File {str(path)} does not exist.")

    return hashlib.sha256(path.read_bytes()).hexdigest()


class AutoRemove:
    def __init__(self, path: Path, recursive: bool = False):
        """
        A small helper that provides a context manager for test binaries, i.e. deletes them after leaving the
        with statement.

        Args:
            path: path to file that should be deleted after context
            recursive: if true and path is a dir, then the whole dir is deleted.

        """
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
