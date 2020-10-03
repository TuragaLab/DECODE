"""
Supplementary code for code examples (mainly jupyter notebook).
"""
import requests
import yaml

import decode


def load_gateway():
    r = requests.get(decode.__gateway__, allow_redirects=True)

    return yaml.load(r.content)
