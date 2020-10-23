# Stuff that handles the example notebooks
import argparse
from typing import Union
from pathlib import Path

from . import examples

try:
    import importlib.resources as pkg_resources
except ImportError:  # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources


def parse_args():
    parser = argparse.ArgumentParser(description='Destination path for example notebooks.')

    parser.add_argument('-p', '--cuda_ix', default=None,
                        help='Specify the cuda device index or set it to false.',
                        type=int, required=False)


def load_examples(path: Union[str, Path]):
    """

    Args:
        path: destination directory

    """

    path = path if isinstance(path, Path) else Path(path)

    for f in ['Introduction.ipynb', 'Evaluation.ipynb', 'Training.ipynb', 'Fit.ipynb']:
        copy_pkg_file(examples, f, path)


def copy_pkg_file(package, file: str, destination: Path):
    """
    Copies a package file to a destination folder.
    """
    template = pkg_resources.read_text(package, file, encoding='utf-8')

    assert destination.is_dir(), "Destination must be directory."
    dest_file = destination / file
    dest_file.write_text(template, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Load example notebooks.")
    parser.add_argument('path', metavar='N', type=str, help='Destination Path')

    load_examples(parser.parse_args().path)
