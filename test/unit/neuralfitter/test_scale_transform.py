import functools
from unittest import mock

import pytest
import torch

import decode.generic.test_utils as t_util
import decode.neuralfitter.scale_transform as scale


class TestSpatialinterpolation:
    @pytest.fixture(scope="class")
    def interp(self):
        return scale.InterpolationSpatial(mode="nearest", scale_factor=2)

    @pytest.fixture(scope="class")
    def interp_new_impl(self):
        impl = functools.partial(
            torch.nn.functional.interpolate, mode="nearest", scale_factor=2
        )
        return scale.InterpolationSpatial(mode=None, impl=impl)

    def test_default_forward(self, interp):
        """
        Tests the forward with the default implementation

        Args:
            interp: fixture as above
        """

        x = torch.rand((1, 1, 32, 32))
        x_out = interp.forward(x)

        assert x.size(-1) * 2 == x_out.size(-1)
        assert x.size(-2) * 2 == x_out.size(-2)
        assert (x_out[0, 0, :2, :2] == x[0, 0, 0, 0]).all()

    @pytest.mark.parametrize(
        "x_in",
        [torch.rand((32, 32)), torch.rand((1, 32, 32)), torch.rand((1, 1, 32, 32))],
    )
    def test_diff_forward(self, interp, x_in):

        out = interp.forward(x_in.clone())

        """Assert"""
        assert x_in.dim() == out.dim()


class TestAmplitudeRescale:
    @pytest.fixture()
    def amp_rescale(self):
        return scale.ScalerAmplitude(scale=1000, offset=5.0)

    def test_rescale_noop(self):
        """
        Tests whether the implementation defaults to no-op when no arguments are specified.

        """

        rescaler = scale.ScalerAmplitude()
        x = torch.rand((2, 3, 64, 64))

        x_out = rescaler.forward(x.clone())

        t_util.tens_almeq(x, x_out)

    def test_rescale(self, amp_rescale):
        x = torch.rand((2, 3, 4, 5))

        assert t_util.tens_almeq(amp_rescale.forward(x.clone()), (x - 5.0) / 1000.0)


class TestTargetRescale:
    @pytest.fixture(scope="class")
    def offset_rescale(self):
        return scale.ScalerOffset(
            scale_x=0.5, scale_y=0.5, scale_z=750.0, scale_phot=10000.0, buffer=1.2
        )

    @pytest.fixture(scope="class")
    def inv_offset_rescale(self, offset_rescale):
        return scale.ScalerInverseOffset(
            scale_x=offset_rescale.sc_x,
            scale_y=offset_rescale.sc_y,
            scale_z=offset_rescale.sc_z,
            scale_phot=offset_rescale.sc_phot,
            buffer=offset_rescale.buffer,
        )

    def test_inverse_rescale(self, inv_offset_rescale):
        """
        Some hard coded testing.

        Args:
            inv_offset_rescale:

        """
        x = torch.zeros(2, 5, 8, 8)
        x[0, 1, 0, 0] = 5000.0
        x[0, 2, 0, 0] = 0.5
        x[0, 3, 0, 0] = -0.4
        x[0, 4, 0, 0] = 500.0
        out = inv_offset_rescale.forward(x)
        assert (out <= 1.0).all()
        assert (out >= -1.0).all()
        assert out[0, 1, 0, 0].item() == pytest.approx(5000 / (10000 * 1.2), 1e-4)
        assert out[0, 2, 0, 0].item() == pytest.approx(0.5 / (0.5 * 1.2), 1e-4)
        assert out[0, 3, 0, 0].item() == pytest.approx(-0.4 / (0.5 * 1.2), 1e-4)
        assert out[0, 4, 0, 0].item() == pytest.approx(500 / (750 * 1.2), 1e-4)

    def test_round(self, inv_offset_rescale, offset_rescale):
        """
        Calculate forth and back to check the values.

        Args:
            inv_offset_rescale:
            offset_rescale:

        """
        x = torch.rand(2, 5, 64, 64)
        inv_derived = offset_rescale.return_inverse()  # derived inverse from offset

        x_hat = offset_rescale.forward(inv_offset_rescale.forward(x))
        assert t_util.tens_almeq(x, x_hat, 1e-6)
        x_hat = offset_rescale.forward(inv_derived.forward(x))
        assert t_util.tens_almeq(x, x_hat, 1e-6)


def test_scaler_tar_list():
    s = scale.ScalerTargetList(2.0, 4.0)
    tar = torch.tensor([2.0, 1.0, 1.0, 4.0])
    mask = mock.MagicMock()
    tar_out, mask_out = s.forward(tar, mask)

    assert tar_out is not tar
    assert mask_out is mask
    assert (tar_out == 1).all()


def test_scaler_model_out():
    s = scale.ScalerModelOutput(2, 4, 8)
    # prob, phot, x, y, z, phot_sig, x_sig, y_sig, z_sig, bg
    x = torch.tensor([1.0, 0.5, 1.0, 1.0, 0.25, 0.5, 1.0, 1.0, 0.25, 0.125])

    out = s.forward(x)
    assert out is not x
