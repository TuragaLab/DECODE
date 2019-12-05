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
        if extension == '.json':
            with open(filename, "w") as write_file:
                json.dump(param, write_file, indent=4)
        elif extension in ('.yml', '.yaml'):
            with open(filename, "w") as yaml_file:
                yaml.dump(param, yaml_file)


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
