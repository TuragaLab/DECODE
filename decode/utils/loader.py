import pathlib
import hashlib
import requests


def check_file(file: (str, pathlib.Path), hash=None):
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


def load(file: (str, pathlib.Path), url: str, hash: str = None) -> None:
    """
    Loads file from URL (and checks hash if present)

    Args:
        file: path where to store downloaded file
        url:
        hash:

    """

    if not isinstance(file, pathlib.Path):
        file = pathlib.Path(file)

    file_www = requests.get(url)
    file_www.raise_for_status()  # raises an error if the file is not available

    with file.open('wb') as f:
        f.write(file_www.content)

    d_hash = hashlib.sha256(file.read_bytes()).hexdigest()
    if hash is not None and not hash == d_hash:
        raise RuntimeError(f"Downloaded file does not match hash.\nSHA-256 of ref.: {hash}\nSHA-256 of downloaded: {d_hash}")


def check_load(file: (str, pathlib.Path), url: str, hash: str = None):
    """
    Loads file freshly when check failes

    Args:
        file:
        url:
        hash:

    """

    if not check_file(file, hash):
        load(file, url, hash)
