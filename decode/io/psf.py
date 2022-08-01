from pathlib import Path
from typing import Union

import mat73
import scipy.io
import torch
from deprecated import deprecated

from ..simulation import psf_kernel
from ..utils import types


def load_spline(
    path: Union[str, Path],
    xextent,
    yextent,
    img_shape,
    **kwargs
) -> psf_kernel.CubicSplinePSF:
    """
    Load spline calibration file. Currently old and new style `.mat` are supported.
    This loader most likely expects calibration files from SMAP.

    Args:
        path: path to calibration file
        xextent: x extent of psf
        yextent: y extent of psf
        img_shape: image shape
        **kwargs: arbitrary kwargs to pass on to `CubicSplinePSF`

    """
    path = path if isinstance(path, Path) else Path(path)

    try:
        calib = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
        calib = types.RecursiveNamespace(**calib).SXY
        coeff = torch.from_numpy(calib.cspline.coeff)

    except NotImplementedError:
        calib = mat73.loadmat(path, use_attrdict=False)
        calib = types.RecursiveNamespace(**calib).SXY
        coeff = torch.from_numpy(calib.cspline.coeff[0])

    ref0 = (
        calib.cspline.x0 - 1,
        calib.cspline.x0 - 1,
        float(calib.cspline.z0),
    )

    dz = calib.cspline.dz

    # necessary because this could be overwritten in kwargs
    if "vx_size" not in kwargs:
        kwargs.update({"vx_size": (1.0, 1.0, dz)})

    if ref0 not in kwargs:
        kwargs.update({"ref0": ref0})

    return psf_kernel.CubicSplinePSF(
        coeff=coeff,
        xextent=xextent,
        yextent=yextent,
        img_shape=img_shape,
        **kwargs
    )


@deprecated(version="0.11.0", reason="use functional interface `load_spline` instead.")
class SMAPSplineCoefficient:
    pass
