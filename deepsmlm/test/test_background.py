import matplotlib.pyplot as plt
import pytest
import torch

import deepsmlm.generic.background as background
import deepsmlm.generic.utils.test_utils as tutil
from deepsmlm.generic.plotting.frame_coord import PlotFrame


class TestBackground:

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
                bg_term = torch.rand_like(x)
                return self.bg_return(xbg=x + bg_term, bg=bg_term)

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
        rframe_bg, bg_term = bgf.forward(rframe)

        assert rframe_bg.size() == rframe.size(), "Wrong shape after background."
        assert bg_term.dim() <= rframe_bg.dim(), "Background term must be of less or equal dimension than input frame."

    def test_additivity(self, bgf, rframe):
        """
        Tests whether the backgorund is additive

        Args:
            bgf: fixture as above
            rframe: fixture as above

        """
        rframe_bg, bg_term = bgf.forward(rframe)
        assert (rframe_bg >= rframe).all(), "Background is not strictly unsigned additive."

        _ = rframe + bg_term  # test whether one can add background to the rframe so that dimension etc. are correct


class TestOofEmitterBackground(TestBackground):

    @pytest.fixture(scope='class')
    def bgf(self):
        extent = ((-0.5, 63.5), (-0.5, 63.5), (-750., 750.))
        img_shape = (64, 64)
        return background.OutOfFocusEmitters(extent[0], extent[1], img_shape, ampl=(1., 1.), num_oof_rg=(1, 1))

    def test_gauss_psf(self, bgf):
        """Tests whether the correct attribute for the gaussian psf is used."""
        assert bgf.gauss_psf.peak_weight

    def test_forward(self, bgf):
        """Test peak heights."""
        x = torch.zeros((1, 1, 64, 64))
        out, _ = bgf.forward(x)
        assert pytest.approx(out.max().item(), 0.01) == 1.


class TestPerlinBg(TestBackground):

    @pytest.fixture(scope='class')
    def bgf(self):
        img_size = (64, 64)
        return background.PerlinBackground(img_size, 2, 20)

    def test_bypass(self, bgf):
        """
        Tests whether what comes in is what comes out if the amplitude is 0.
        :param perlin_default:
        :return:
        """
        bgf.amplitude = 0.
        x_in = torch.zeros((2, 3, 64, 64))
        x_out, _ = bgf.forward(x_in.clone())
        assert tutil.tens_almeq(x_in, x_out)

    def test_approximate_range(self, bgf):
        x_in = torch.zeros((2, 3, 64, 64))
        x_out, _ = bgf.forward(x_in.clone())
        assert x_out.min() >= 0.
        assert x_out.max() <= bgf.amplitude * 1.


class TestMultiPerlin(TestBackground):

    @pytest.fixture(scope='class')
    def bgf(self):
        return background.MultiPerlin((64, 64), [64, 32, 16, 8], [1, 1, 1, 1],
                                      norm_amps=False,
                                      draw_amps=True)

    @pytest.mark.skip_plot
    def test_multiscale(self, bgf):
        out, _ = bgf.forward(torch.zeros((2, 3, 64, 64)))
        PlotFrame(out[0, 0]).plot()
        plt.colorbar()
        plt.show()
