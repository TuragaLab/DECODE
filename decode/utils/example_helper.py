"""
Supplementary code for code examples (mainly jupyter notebook).
"""
import requests
import yaml
import zipfile

import decode
from decode.utils import loader


def load_gateway():
    r = requests.get(decode.__gateway__, allow_redirects=True)

    return yaml.load(r.content)


def load_example_package(path, url, hash):

    if not loader.check_file(path, hash):
        loader.load(path, url, hash)

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall()

    else:
        print("Found file already in Cache.")

    tif_path = path / 'frames.tiff'
    model_path = path / 'model.pt'
    param_path = path / 'param_run.yaml'

    return tif_path, model_path, param_path