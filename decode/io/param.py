from pathlib import Path
from typing import Any, IO, Union

try:
    import importlib.resources as pkg_resources
except ImportError:  # backported to py ver <37 `importlib_resources`.
    import importlib_resources as pkg_resources


import pydantic
from omegaconf import OmegaConf, DictConfig


def _load(p: Union[str, Path, IO[Any]]) -> DictConfig:
    return OmegaConf.load(p)


def load_reference() -> DictConfig:
    """Loads reference config."""
    from ..utils import reference_files

    param_ref = pkg_resources.open_text(reference_files, "reference.yaml")
    param_ref = _load(param_ref)

    return param_ref


@pydantic.validate_arguments
def copy_reference(path: pydantic.DirectoryPath) -> tuple[Path, Path]:
    """
    Copies reference config to path

    Args:
        path: destination dir

    """
    from ..utils import reference_files

    ref = pkg_resources.read_text(reference_files, "reference.yaml")
    friendly = pkg_resources.read_text(reference_files, "param_friendly.yaml")

    path_ref = path / "reference.yaml"
    path_friendly = path / "param_friendly.yaml"

    path_ref.write_text(ref)
    path_friendly.write_text(friendly)

    return path_ref, path_friendly
