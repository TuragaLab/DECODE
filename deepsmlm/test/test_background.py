import torch
import pytest
import matplotlib.pyplot as plt
import deepsmlm.generic.background as background
from deepsmlm.generic.plotting.frame_coord import PlotFrame


class TestExperimentBg:

    @pytest.fixture(scope='class')
    def exp_bg(self):
        extent = ((-0.5, 63.5), (-0.5, 63.5), (-750., 750.))
        img_shape = (64, 64)
        return background.OutOfFocusEmitters(extent[0], extent[1], img_shape, bg_range=(1., 1.), num_bg_emitter=1)

    def test_gauss_psf(self, exp_bg):
        """Tests whether the correct attribute for the gaussian psf is used."""
        assert exp_bg.psf.peak_weight

    def test_forward(self, exp_bg):
        """Test peak heights."""
        x = torch.zeros((1, 1, 64, 64))
        out = exp_bg.forward(x)
        assert pytest.approx(out.max().item(), 0.01) == 1.


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
