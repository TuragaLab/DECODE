from dataclasses import dataclass
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Union, Optional, List
import yaml

from . import param_io


@dataclass
class _InferenceAtom:
    frame_path: Union[str, Path] = None
    model_path: Union[str, Path] = None
    output_path: Union[str, Path] = None
    param: param_io.RecursiveNamespace = None


def _load_default_cfg(cfg: param_io.RecursiveNamespace) -> _InferenceAtom:
    param = param_io.autofill_dict(
        cfg.param.to_dict(),
        param_io.load_params(cfg.param_path, autofill=False).to_dict()
    )
    param = param_io.RecursiveNamespace(**param)

    return _InferenceAtom(
        frame_path=None,
        model_path=cfg.model_path,
        param=param,
    )


def get_file_list(path: Union[str, Path], suffix: Optional = None, pattern: Optional = None) -> List[Path]:
    """Get files from path"""
    if not isinstance(path, Path):
        path = Path(path)

    if path.is_file():
        return [path]

    if suffix is None and pattern is None:
        raise ValueError("Specify either suffix XOR pattern.")

    if suffix is not None:
        pattern = f"*{suffix}"

    return sorted(path.rglob(pattern))


def compile_fit(cfg) -> _InferenceAtom:

    # get defaults


    # loop over files
    frame_files = []
    for p in cfg.inferences:
        # todo add suffix etc.
        frame_files.append(p.path)

    # loop over dirs
