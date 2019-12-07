import json
import yaml
import dotmap
import pathlib
import warnings


class ParamHandling:

    file_extensions = ('.json', '.yml', '.yaml')

    def __init__(self):

        self.params_dict = None
        self.params_dot = None

    def _check_return_extension(self, filename):
        """
        Checks the specified file suffix
        :param filename:
        :return:
        """
        extension = pathlib.PurePath(filename).suffix
        if extension not in self.file_extensions:
            raise ValueError(f"Filename must be in {self.file_extensions}")

        return extension

    def load_params(self, filename):
        """
        Loads parameters from .json or .yml
        :param filename:
        :return:
        """
        extension = self._check_return_extension(filename)
        if extension == '.json':
            with open(filename) as json_file:
                params_dict = json.load(json_file)
        elif extension in ('.yml', '.yaml'):
            with open(filename) as yaml_file:
                params_dict = yaml.safe_load(yaml_file)

        params_dot = dotmap.DotMap(params_dict)

        self.params_dict = params_dict
        self.params_dot = params_dot

        return params_dot

    def write_params(self, filename, param):
        extension = self._check_return_extension(filename)
        param = param.toDict()

        """Create Folder if not exists."""
        p = pathlib.Path(filename)
        try:
            pathlib.Path(p.parents[0]).mkdir(parents=False, exist_ok=True)
        except FileNotFoundError:
            raise FileNotFoundError("I will only create the last folder for parameter saving. "
                                    "But the path you specified lacks more folders or is completely wrong.")

        if extension == '.json':
            with open(filename, "w") as write_file:
                json.dump(param, write_file, indent=4)
        elif extension in ('.yml', '.yaml'):
            with open(filename, "w") as yaml_file:
                yaml.dump(param, yaml_file)

    def convert_param_file(self, file_in, file_out):
        """
        Simple Wrapper to change yml to json or the other way around.
        :param file_in:
        :param file_out:
        :return:
        """
        params = self.load_params(file_in)
        self.write_params(file_out, params)

    @staticmethod
    def convert_param_debug(param):
        param.HyperParameter.pseudo_ds_size = 1024
        param.HyperParameter.test_size = 128
        param.InOut.model_out = 'network/debug.pt'


def write_params(filename, param):
    warnings.warn(
        "write_params function is deprecated. Will call ParamHandling for you instead. Removed soon.",
        DeprecationWarning
    )
    return ParamHandling().write_params(filename, param)


def load_params(filename):
    warnings.warn(
        "load_params function is deprecated. Will call ParamHandling for you instead. Removed soon.",
        DeprecationWarning
    )
    return ParamHandling().load_params(filename)
