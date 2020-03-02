import torch
import pytest
import matplotlib.pyplot as plt
import deepsmlm.generic.background as background
from deepsmlm.generic.plotting.frame_coord import PlotFrame

import deepsmlm.generic.utils.test_utils as tutil


class TestBackgroundAbc:

    @pytest.fixture(scope='class')
    def bgf(self):
        """
        Dummy background fixture.

        Returns:

        """
        class DummyBG(background.Background):
            def __init__(self, xextent, yextent, img_shape):
                super().__init__()

                self.xextent = xextent
                self.yextent = yextent
                self.img_shape = img_shape

            def forward(self, x: torch.Tensor):
                return x + torch.rand_like(x)

        xextent = (-0.5, 63.5)
        yextent = (-0.5, 63.5)
        img_shape = (64, 64)

        return DummyBG(xextent, yextent, img_shape)

    @pytest.fixture(scope='class')
    def rframe(self):
        """
        Just a random frame batch.

        Returns:
            rframe (torch.Tensor): random frame batch

        """
        return torch.rand((2, 3, 64, 64))

    def test_shape(self, bgf, rframe):
        """
        Tests shape equality

        Args:
            bgf: fixture as above
            rframe: fixture as above

        """
        assert bgf.forward(rframe).size() == rframe.size(), "Wrong shape after background."

    def test_additivity(self, bgf, rframe):
        """
        Tests whether the backgorund is additive

        Args:
            bgf: fixture as above
            rframe: fixture as above

        """
        assert (bgf.forward(rframe) >= rframe).all(), "Background is not strictly unsigned additive."


class TestOofEmitterBackground(TestBackgroundAbc):

    @pytest.fixture(scope='class')
    def bgf(self):
        extent = ((-0.5, 63.5), (-0.5, 63.5), (-750., 750.))
        img_shape = (64, 64)
        return background.OutOfFocusEmitters(extent[0], extent[1], img_shape, ampl=(1., 1.), num_oof_rg=[1, 2])

    def test_gauss_psf(self, bgf):
        """Tests whether the correct attribute for the gaussian psf is used."""
        assert bgf.gauss_psf.peak_weight

    def test_forward(self, bgf):
        """Test peak heights."""
        x = torch.zeros((1, 1, 64, 64))
        out = bgf.forward(x)
        assert pytest.approx(out.max().item(), 0.01) == 1.


class TestPerlinBg(TestBackgroundAbc):

    @pytest.fixture(scope='class')
    def perlin_default(self):
        img_size = (64, 64)
        return background.PerlinBackground(img_size, 2, 20)

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

    # @pytest.mark.skip('Test MultiScale only for plotting.')
    def test_multiscale(self):
        img_size = (64, 64)
        cand = background.PerlinBackground.multi_scale_init(img_size, [64, 32, 16, 8], [1, 1, 1, 1],
                                                            norm_amplitudes=False,
                                                            draw_amps=True)
        out = cand.forward(torch.zeros((2, 3, 64, 64)))
        PlotFrame(out[0, 0]).plot()
        plt.colorbar()
        plt.show()

@pytest.mark.skip("Only for checking the plotting.")
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