import functools
import pytest
import torch

import deepsmlm.neuralfitter.scale_transform as scf
import deepsmlm.generic.utils.test_utils as t_util


class TestSpatialinterpolation:

    @pytest.fixture(scope='class')
    def interp(self):
        return scf.SpatialInterpolation(dim=2, mode='nearest', scale_factor=2)

    @pytest.fixture(scope='class')
    def interp_new_impl(self):
        impl = functools.partial(torch.nn.functional.interpolate, mode='nearest', scale_factor=2)
        return scf.SpatialInterpolation(mode=None, impl=impl)

    @pytest.mark.parametrize("x", [torch.rand((32, 32)), torch.rand((1, 32, 32))])
    def test_dimensionality_test(self, interp, x):
        """
        Tests the dimensionality assertions

        Args:
            interp: fixture as above
            x: tensors that are of the wrong dimensionality and should lead to fail

        """

        with pytest.raises(ValueError):
            interp.forward(x)

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


class TestTargetRescale:

    @pytest.fixture(scope='class')
    def offset_rescale(self):
        return scf.OffsetRescale(0.5, 0.5, 750., 10000., 1.2)

    @pytest.fixture(scope='class')
    def inv_offset_rescale(self, offset_rescale):
        return scf.InverseOffsetRescale(offset_rescale.sc_x,
                                        offset_rescale.sc_y,
                                        offset_rescale.sc_z,
                                        offset_rescale.sc_phot,
                                        offset_rescale.buffer)

    def test_inverse_rescale(self, inv_offset_rescale):
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
        x = torch.rand(2, 5, 64, 64)
        x_hat = offset_rescale.forward(inv_offset_rescale.forward(x))
        assert t_util.tens_almeq(x, x_hat, 1e-6)