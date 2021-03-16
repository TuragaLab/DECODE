import matplotlib.pyplot as plt
import pytest
import torch

from decode.generic import emitter
from decode.plot import PlotFrameCoord
from decode.renderer import renderer


class TestRenderer2D:

    @pytest.fixture()
    def rend(self):
        return renderer.Renderer2D(
            plot_axis=(0, 1), xextent=(0., 100.), yextent=(0., 100.),
            px_size=10., sigma_blur=10., rel_clip=None, abs_clip=None)

    @pytest.fixture()
    def em(self):
        """Setup"""
        xyz = torch.tensor([[10., 50., 100.]])
        em = emitter.CoordinateOnlyEmitter(xyz, xy_unit='nm')
        em.phot = torch.ones_like(em.phot)

        return em

    def test_forward(self, rend, em):
        histogram = rend.forward(em)

        assert histogram.size() == torch.Size([10, 10])
        assert histogram[1, 5] != 0  # Todo: Could also be histogram[5, 1] but this fails as well
        assert histogram.abs().sum() == histogram[1, 5]

    @pytest.mark.plot
    def test_plot_frame_render_visual(self, rend, em):
        PlotFrameCoord(torch.zeros((101, 101)), em.xyz_nm).plot()
        plt.show()

        rend.render(em)
        plt.show()


class TestRendererIndividual2D:

    @pytest.fixture()
    def rend(self):
        return renderer.RendererIndividual2D(
            plot_axis = (0,1), xextent=(0., 100.), yextent=(0., 100.),
            zextent=(-100., 100.), colextent=(0.,100.), px_size=10.,
            filt_size=20, rel_clip=None, abs_clip=None)

    @pytest.fixture()
    def em(self):
        """Setup"""
        xyz = torch.rand(100, 3) * torch.Tensor([[100., 100., 1000.]])
        return emitter.EmitterSet(xyz, xyz_sig=xyz*0.1, phot=torch.ones(100), frame_ix=torch.arange(100), xy_unit='nm')

    def test_forward(self, rend, em):
        histogram = rend.forward(em, torch.arange(len(em)))

        assert histogram.size() == torch.Size([10, 10, 3])
        assert histogram.sum() > 0.

    @pytest.mark.plot
    def test_plot_frame_render_visual(self, rend, em):
        PlotFrameCoord(torch.zeros((101, 101)), em.xyz_nm).plot()
        plt.show()

        rend.render(em, torch.arange(len(em)))
        plt.show()
