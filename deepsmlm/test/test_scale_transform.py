import functools

import pytest
import torch

import deepsmlm.generic.test_utils as t_util
import deepsmlm.neuralfitter.scale_transform as scf


class TestSpatialinterpolation:

    @pytest.fixture(scope='class')
    def interp(self):
        return scf.SpatialInterpolation(mode='nearest', scale_factor=2)

    @pytest.fixture(scope='class')
    def interp_new_impl(self):
        impl = functools.partial(torch.nn.functional.interpolate, mode='nearest', scale_factor=2)
        return scf.SpatialInterpolation(mode=None, impl=impl)

    def test_default_forward(self, interp):
        """
        Tests the forward with the default implementation

        Args:
            interp: fixture as above
        """

        x = torch.rand((1, 1, 32, 32))
        x_out = interp.forward(x)

        """Dimension"""
        assert x.size(-1) * 2 == x_out.size(-1)
        assert x.size(-2) * 2 == x_out.size(-2)

        """Values"""
        assert (x_out[0, 0, :2, :2] == x[0, 0, 0, 0]).all()

    @pytest.mark.parametrize("x_in", [torch.rand((32, 32)), torch.rand((1, 32, 32)), torch.rand((1, 1, 32, 32))])
    def test_diff_forward(self, interp, x_in):

        out = interp.forward(x_in.clone())

        """Assert"""
        assert x_in.dim() == out.dim()


class TestAmplitudeRescale:

    @pytest.fixture()
    def amp_rescale(self):
        return scf.AmplitudeRescale(norm_value=1000)

    def test_rescale(self, amp_rescale):
        x = torch.rand((2, 3, 4, 5))

        assert t_util.tens_almeq(amp_rescale.forward(x), x / 1000.)


class TestTargetRescale:

    @pytest.fixture(scope='class')
    def offset_rescale(self):
        return scf.OffsetRescale(scale_x=0.5, scale_y=0.5, scale_z=750., scale_phot=10000., buffer=1.2)

    @pytest.fixture(scope='class')
    def inv_offset_rescale(self, offset_rescale):
        return scf.InverseOffsetRescale(scale_x=offset_rescale.sc_x,
                                        scale_y=offset_rescale.sc_y,
                                        scale_z=offset_rescale.sc_z,
                                        scale_phot=offset_rescale.sc_phot,
                                        buffer=offset_rescale.buffer)

    def test_inverse_rescale(self, inv_offset_rescale):
        """
        Some hard coded testing.

        Args:
            inv_offset_rescale:

        """
        x = torch.zeros(2, 5, 8, 8)
        x[0, 1, 0, 0] = 5000.
        x[0, 2, 0, 0] = 0.5
        x[0, 3, 0, 0] = -0.4
        x[0, 4, 0, 0] = 500.
        out = inv_offset_rescale.forward(x)
        assert (out <= 1.).all()
        assert (out >= -1.).all()
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
        """Setup"""
        x = torch.rand(2, 5, 64, 64)
        inv_derived = offset_rescale.return_inverse()  # derived inverse from offset

        """Run"""
        x_hat = offset_rescale.forward(inv_offset_rescale.forward(x))
        assert t_util.tens_almeq(x, x_hat, 1e-6)
        x_hat = offset_rescale.forward(inv_derived.forward(x))
        assert t_util.tens_almeq(x, x_hat, 1e-6)



