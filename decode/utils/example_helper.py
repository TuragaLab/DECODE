"""
Supplementary code for code examples (mainly jupyter notebook). Some of this seams utterly less abstract and hard-coded, but it is a dedicated example helper ...

"""
import requests
import pathlib
import yaml
import zipfile

import decode
from decode.utils import loader


def load_gateway():
    r = requests.get(decode.__gateway__, allow_redirects=True)

    return yaml.load(r.content, Loader=yaml.FullLoader)


def load_example_package(path: pathlib.Path, url: str, hash: str, mode: str):
    """

    Args:
        path: destination where to save example package
        url:
        hash: sha 256 hash
        mode: 'fit' or 'train'

    """

    zip_folder = path.parent / path.stem

    if not loader.check_file(path, hash):
        loader.load(path, url, hash)

        zip_folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path=zip_folder)

    else:
        print("Found file already in Cache.")

    return zip_folder
