import pytest

import torch
import numpy as np
import matplotlib.pyplot as plt

from decode.generic import emitter
from decode.renderer import renderer
from decode.plot import PlotFrameCoord


class TestRenderer2D:

    @pytest.fixture()
    def rend(self):
        return renderer.Renderer2D(xextent=(0., 100.), yextent=(0., 100.), px_size=10., sigma_blur=10.,
                                   clip_percentile=None)

    def test_plot_frame_render_equality(self, rend):

        """Setup"""
        xyz = torch.tensor([[10., 50., 100.]])
        em = emitter.CoordinateOnlyEmitter(xyz, xy_unit='nm')

        """Run"""
        PlotFrameCoord(torch.zeros((101, 101)), em.xyz_nm).plot()
        plt.show()

        rend.render(em)
        plt.show()
