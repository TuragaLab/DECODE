import torch
import pytest
import matplotlib.pyplot as plt
import deepsmlm.generic.background as background
from deepsmlm.generic.plotting.frame_coord import PlotFrame
from deepsmlm.generic.utils.processing import TransformSequence

import deepsmlm.test.utils_ci as tutil

class TestExperimentBg:

    @pytest.fixture(scope='class')
    def exp_bg(self):
        extent = ((-0.5, 63.5), (-0.5, 63.5), (-750., 750.))
        img_shape = (64, 64)
        return background.OutOfFocusEmitters(extent[0], extent[1], img_shape, bg_range=(1., 1.), num_bgem_range=[1, 1])

    def test_gauss_psf(self, exp_bg):
        """Tests whether the correct attribute for the gaussian psf is used."""
        assert exp_bg.psf.peak_weight

    def test_forward(self, exp_bg):
        """Test peak heights."""
        x = torch.zeros((1, 1, 64, 64))
        out = exp_bg.forward(x)
        assert pytest.approx(out.max().item(), 0.01) == 1.


class TestPerlinBg:

    @pytest.fixture(scope='class')
    def perlin_default(self):
        img_size = (64, 64)
        return background.PerlinBackground(img_size, (2, 2), 20)

    def test_bypass(self, perlin_default):
        """
        Tests whether what comes in is what comes out if the amplitude is 0.
        :param perlin_default:
        :return:
        """
        perlin_default.amplitude = 0.
        x_in = torch.zeros((2, 3, 64, 64))
        x_out = perlin_default.forward(x_in.clone())
        assert tutil.tens_almeq(x_in, x_out)

    def test_approximate_range(self, perlin_default):
        x_in = torch.zeros((2, 3, 64, 64))
        x_out = perlin_default.forward(x_in.clone())
        assert x_out.min() >= 0.
        assert x_out.max() <= perlin_default.amplitude * 1.

    def test_multiscale(self):
        img_size = (64, 64)
        cand = background.PerlinBackground.multi_scale_init(img_size, [[1, 1], [2, 2], [4, 4]], [100, 50, 20])
        out = cand.forward(torch.zeros((2, 3, 64, 64)))
        PlotFrame(out[0, 0]).plot()
        plt.colorbar()
        plt.show()

    @pytest.mark.skip("Only for checking the plotting.")
    def test_plotting(self, perlin_default):

        x = torch.zeros((2, 3, 64, 64))
        out = perlin_default.forward(x)
        plt.figure(figsize=(12, 12))
        PlotFrame(out[0, 0]).plot()
        plt.colorbar()
        plt.show()


def test_nonuniformbg():

    bg_u = background.NonUniformBackground(0.1, img_size=(32, 32))
    x = torch.rand((2, 3, 32, 32)) * 4

    x_out = bg_u.forward(x)

    plt.figure(figsize=(12, 12))
    PlotFrame(x[0, 0]).plot()
    plt.show()
    plt.figure(figsize=(12, 12))
    PlotFrame(x_out[0, 0]).plot()
    plt.show()