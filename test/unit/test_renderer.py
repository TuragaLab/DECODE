import copy

import matplotlib.pyplot as plt
import pytest
import torch

from decode.generic import emitter
from decode.generic import test_utils
from decode.plot import PlotFrameCoord
from decode.renderer import renderer


class TestRenderer2D:

    @pytest.fixture()
    def rend(self):
        return renderer.Renderer2D(
            plot_axis=(0, 1), xextent=(0., 100.), yextent=(0., 100.),
            px_size=10., sigma_blur=None, rel_clip=None, abs_clip=None)

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
        assert histogram[1, 5] != 0
        assert histogram.sum() == histogram[1, 5]

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
            zextent=(0.,1000.), colextent=(0.,100.), px_size=10.,
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_forward(self, rend, em):
        rend_cpu = copy.deepcopy(rend)
        rend_cuda = rend
        rend_cuda.device = 'cuda:0'

        assert test_utils.tens_almeq(
            rend_cuda.forward(em, torch.arange(len(em))),
            rend_cpu.forward(em, torch.arange(len(em))),
            1e-4)

    @pytest.mark.plot
    def test_plot_frame_render_visual(self, rend, em):
        PlotFrameCoord(torch.zeros((101, 101)), em.xyz_nm).plot()
        plt.show()

        rend.render(em, torch.arange(len(em)))
        plt.show()
