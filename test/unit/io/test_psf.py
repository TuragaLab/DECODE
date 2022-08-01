from unittest import mock

import numpy as np
import pytest

from decode import io
from decode.simulation import psf_kernel


@pytest.fixture
def _spline_calib():
    calib = mock.MagicMock()
    calib["SXY"].cspline.coeff = np.random.rand(17, 18, 19, 20)
    calib["SXY"].cspline.x0 = 6
    calib["SXY"].cspline.z0 = 17
    calib["SXY"].cspline.dz = 10

    return calib


@mock.patch("decode.io.psf.scipy.io.loadmat")
@mock.patch("decode.io.psf.mat73.loadmat")
@pytest.mark.parametrize("mat_style", ["old", "new"])
def test_load_spline(mock_mat73, mock_sci, mat_style, _spline_calib):
    p = mock.MagicMock()

    if mat_style == "old":
        mock_sci.return_value = _spline_calib
    elif mat_style == "new":
        mock_sci.side_effect = NotImplementedError
        mock_mat73.return_value = _spline_calib

    psf = io.psf.load_spline(p, [0., 17.], [0., 18.], [17, 18])
    assert isinstance(psf, psf_kernel.CubicSplinePSF)
