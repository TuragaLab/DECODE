import json
# import copy
import pathlib
from pathlib import Path
from typing import Union
try:
    import importlib.resources as pkg_resources
except ImportError:  # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

import yaml

from .types import RecursiveNamespace


class ParamHandling:
    file_extensions = ('.json', '.yml', '.yaml')

    def __init__(self):

        self.params_dict = None
        self.params_dot = None

    def _check_return_extension(self, filename):
        """
        Checks the specified file suffix as to whether it is in the allowed list

        Args:
            filename

        """
        extension = pathlib.PurePath(filename).suffix
        if extension not in self.file_extensions:
            raise ValueError(f"Filename must be in {self.file_extensions}")

        return extension

    def load_params(self, filename: str) -> RecursiveNamespace:
        """
        Load parameters from file

        Args:
            filename:

        Returns:

        """

        extension = self._check_return_extension(filename)
        if extension == '.json':
            with open(filename) as json_file:
                params_dict = json.load(json_file)

        elif extension in ('.yml', '.yaml'):
            with open(filename) as yaml_file:
                params_dict = yaml.safe_load(yaml_file)

        params_ref = load_reference()
        params_dict = autofill_dict(params_dict, params_ref)
        params = RecursiveNamespace(**params_dict)

        self.params_dict = params_dict
        self.params_dot = params

        return params

    def write_params(self, filename: Union[str, pathlib.Path], param: Union[dict, RecursiveNamespace]):
        """
        Write parameter file to path

        Args:
            filename:
            param:

        """
        filename = filename if isinstance(filename, pathlib.Path) else pathlib.Path(filename)

        extension = self._check_return_extension(filename)

        if isinstance(param, RecursiveNamespace):
            param = param.to_dict()

        """Create Folder if not exists."""
        p = pathlib.Path(filename)
        try:
            pathlib.Path(p.parents[0]).mkdir(parents=False, exist_ok=True)
        except FileNotFoundError:
            raise FileNotFoundError("I will only create the last folder for parameter saving. "
                                    "But the path you specified lacks more folders or is completely wrong.")

        if extension == '.json':
            with filename.open('w') as write_file:
                json.dump(param, write_file, indent=4)
        elif extension in ('.yml', '.yaml'):
            with filename.open('w') as yaml_file:
                yaml.dump(param, yaml_file)

    def convert_param_file(self, file_in, file_out):
        """
        Simple wrapper to convert file from and to json / yaml
        """

        params = self.load_params(file_in)
        self.write_params(file_out, params)

    @staticmethod
    def convert_param_debug(param):
        param.HyperParameter.pseudo_ds_size = 1024
        param.TestSet.test_size = 128
        param.InOut.model_out = 'network/debug.pt'


def load_params(file):  # alias
    return ParamHandling().load_params(file)


def load_reference() -> dict:
    """
    Loads the static reference .yaml file because there we have the full sets and default values.

    """

    from . import reference_files
    param_ref = pkg_resources.open_text(reference_files, 'reference.yaml')
    param_ref = yaml.load(param_ref, Loader=yaml.SafeLoader)

    return param_ref


def autofill_dict(x: dict, reference: dict, mode_missing: str = 'include') -> dict:
    """
    Fill dict `x` with keys and values of reference if they are not present in x.

    Args:
        x: input dict to be filled
        reference: reference dictionary
        mode_missing:

    """

    if mode_missing == 'exclude':  # create new dict and copy
        out = {}
    elif mode_missing == 'include':
        out = x
    else:
        raise ValueError

    for k, v in reference.items():
        if isinstance(v, dict):
            out[k] = autofill_dict(x[k] if k in x else {}, v)
        elif k in x:  # below here never going to be a dict
            out[k] = x[k]
        else:
            out[k] = reference[k]

    return out


def save_params(file, param):  # alias
    ParamHandling().write_params(file, param)


def autoset_scaling(param):
    def set_if_none(var, value):
        if var is None:
            var = value
        return var

    param.Scaling.input_scale = set_if_none(param.Scaling.input_scale, param.Simulation.intensity_mu_sig[0] / 50)
    param.Scaling.phot_max = set_if_none(param.Scaling.phot_max,
                                         param.Simulation.intensity_mu_sig[0] + 8 * param.Simulation.intensity_mu_sig[
                                             1])

    param.Scaling.z_max = set_if_none(param.Scaling.z_max, param.Simulation.emitter_extent[2][1] * 1.2)
    if param.Scaling.input_offset is None:
        if isinstance(param.Simulation.bg_uniform, (list, tuple)):
            param.Scaling.input_offset = (param.Simulation.bg_uniform[1] + param.Simulation.bg_uniform[0]) / 2
        else:
            param.Scaling.input_offset = param.Simulation.bg_uniform

    if param.Scaling.bg_max is None:
        if isinstance(param.Simulation.bg_uniform, (list, tuple)):
            param.Scaling.bg_max = param.Simulation.bg_uniform[1] * 1.2
        else:
            param.Scaling.bg_max = param.Simulation.bg_uniform * 1.2

    return param


def add_root_relative(path: (str, Path), root: (str, Path)):
    """
    Adds the root to a path if the path is not absolute

    Args:
        path (str, Path): path to file
        root (str, Path): root path

    Returns:
        Path: absolute path to file

    """
    if not isinstance(path, Path):
        path = Path(path)

    if not isinstance(root, Path):
        root = Path(root)

    if path.is_absolute():
        return path

    else:
        return root / path


def copy_reference_param(path: Union[str, Path]):
    """
    Copies our param references to the desired path

    Args:
        path: destination path, must exist and be a directory

    """

    path = path if isinstance(path, Path) else Path(path)
    assert path.is_dir()

    from . import reference_files
    pr = pkg_resources.read_text(reference_files, "reference.yaml")
    pf = pkg_resources.read_text(reference_files, "param_friendly.yaml")

    (path / "reference.yaml").write_text(pr)
    (path / "param_friendly.yaml").write_text(pf)

