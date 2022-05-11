"""Provides means to handle test assets, i.e. check if file exists and loads."""

import pathlib

import click
import yaml

from decode import utils


class AssetHandler:
    asset_list_path = pathlib.Path(__file__).resolve().parents[2] / "test/assets/asset_list.yaml"

    def __init__(self):

        with self.asset_list_path.open() as f:
            self.dict = yaml.safe_load(f)

        self._names = [d['name'] for d in self.dict]

    def auto_load(self, file: (str, pathlib.Path)):
        if not isinstance(file, pathlib.Path):
            file = pathlib.Path(file)

        # check that file is in list and get index
        ix = self._names.index(str(file.name))

        utils.files.check_load(file, url=self.dict[ix]['url'], hash=self.dict[ix]['hash'])


@click.command()
@click.option("--file", required=True, help="Specify a file that should be hased")
def hash_file_cmd(file: str):
    """
    Wrapper function to make this script callable from command line to hash new files.
    It will print the result to the console. Will treat all files in byte mode.

    Args:
        file (str): path of file

    """
    print(utils.files.hash_file(pathlib.Path(file)))


if __name__ == '__main__':
    hash_file_cmd()
