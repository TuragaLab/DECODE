from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, List

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
        output_path=cfg.output_suffix,  # this is a slight abuse of notation
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


def compile_fit(cfg: param_io.RecursiveNamespace) -> List[_InferenceAtom]:
    # get default
    default = _load_default_cfg(cfg)

    # ToDo: flatten directories to files

    # loop over files
    inferences = []
    for p in cfg.inferences:
        p = p.to_dict()

        path = Path(p.pop('path'))
        if not path.is_file():
            raise FileNotFoundError(f"File in path {path} not found.")

        if len(p) == 0:
            continue

        if len(p) >= 1 or 'param' not in p:
            raise NotImplementedError("Currently it is only supported to overwrite params. "
                                      "Not model_path or param_path.")

        param = param_io.RecursiveNamespace(
            **param_io.autofill_dict(
                p.pop('param'),
                default.param.to_dict()
            )
        )

        inferences.append(_InferenceAtom(
            frame_path=path,
            model_path=default.model_path,
            output_path=path.with_name(path.stem + '_decode_fit' + default.output_path),
            param=param,
        ))

    return inferences
