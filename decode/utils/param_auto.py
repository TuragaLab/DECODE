from ..io import param
from typing import Optional, Sequence
from . import types


def _autofill_dict(x: dict, reference: dict, mode_missing: str = "include") -> dict:
    """
    Fill dict `x` with keys and values of reference if they are not present in x.

    Args:
        x: dict to be filled
        reference: reference dict
        mode_missing: what to do if there are values in x that are not present in ref
         `exclude` means that key-value pairs present in x but not in ref will not be
         part of the output dict.
         `include` means that they are.
         `raise` will raise an error if x contains more keys than reference does
    """
    out = dict()
    if mode_missing == "exclude":  # create new dict and copy from ref
        pass
    elif mode_missing == "include":
        out = x
    elif mode_missing == "raise":
        if not set(x.keys()).issubset(set(reference.keys())):
            raise ValueError("There are more keys in `x` than in `reference`.")
    else:
        raise ValueError(f"Not supported mode_missing type: {mode_missing}")

    for k, v in reference.items():
        if isinstance(v, dict):
            out[k] = _autofill_dict(x[k] if k in x else {}, v)
        elif k in x:  # below here never going to be a dict
            out[k] = x[k]
        else:
            out[k] = reference[k]

    return out


def auto_scale(cfg):
    """
    Automatically determine scale from simulation parameters if not manually set

    Args:
        cfg:

    Returns:

    """

    def set_if_none(var, value):
        if var is None:
            var = value
        return var

    cfg.Scaling.input.frame.scale = set_if_none(
        cfg.Scaling.input.frame.scale, cfg.Simulation.intensity.mean / 50
    )
    cfg.Scaling.output.phot.max = set_if_none(
        cfg.Scaling.output.phot.max,
        cfg.Simulation.intensity.mean + 8 * cfg.Simulation.intensity.std,
    )

    cfg.Scaling.output.z.max = set_if_none(
        cfg.Scaling.output.z.max, cfg.Simulation.emitter_extent.z[1] * 1.2
    )
    if cfg.Scaling.input.frame.offset is None:
        if isinstance(cfg.Simulation.bg[0].uniform, Sequence):
            cfg.Scaling.input.frame.offset = (
                cfg.Simulation.bg[0].uniform[1] + cfg.Simulation.bg[0].uniform[0]
            ) / 2
        else:
            cfg.Scaling.input.frame.offset = cfg.Simulation.bg[0].uniform[0]

    if cfg.Scaling.output.bg.max is None:
        if isinstance(cfg.Simulation.bg[0].uniform, Sequence):
            cfg.Scaling.output.bg.max = cfg.Simulation.bg[0].uniform[1] * 1.2
        else:
            cfg.Scaling.output.bg.max = cfg.Simulation.bg[0].uniform * 1.2

    return cfg


class AutoConfig:
    def __init__(
        self,
        fill: bool = True,
        fill_test: bool = True,
        auto_scale: bool = True,
        ref: Optional[dict] = None,
    ):
        """
        Automate config handling

        Args:
            fill: fill missing values by reference
            fill_test: fill test set values by training set / simulation
            auto_scale: infer scaling parameters
            ref: reference dict for automatic filling
        """
        self._do_fill = fill
        self._do_fill_test = fill_test
        self._do_auto_scale = auto_scale
        self._reference = ref if ref is not None else dict(**param.load_reference())

    def parse(self, cfg: dict) -> dict:
        cfg = self._fill(cfg) if self._do_fill else cfg
        cfg = self._fill_test(cfg) if self._do_fill_test else cfg
        cfg = self._auto_scale(cfg) if self._do_auto_scale else cfg

        return cfg

    def _fill(self, cfg: dict) -> dict:
        # fill config by reference
        return _autofill_dict(cfg, self._reference, mode_missing="raise")

    def _fill_test(self, cfg: dict) -> dict:
        # fill test set by training set config
        cfg["Test"] = _autofill_dict(
            cfg["Test"], cfg["Simulation"], mode_missing="raise"
        )
        return cfg

    def _auto_scale(self, cfg: dict) -> dict:
        # fill scaling parameters by simulation
        cfg = types.RecursiveNamespace(**cfg)
        return dict(auto_scale(cfg))
