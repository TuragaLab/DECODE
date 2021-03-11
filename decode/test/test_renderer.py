import matplotlib.pyplot as plt
import pytest
import torch

from decode.generic import emitter
from decode.plot import PlotFrameCoord
from decode.renderer import renderer


class TestRenderer2D:

    @pytest.fixture()
    def rend(self):
        return renderer.Renderer2D(plot_axis = (0,1), xextent=(0., 100.), yextent=(0., 100.), px_size=10., sigma_blur=10.,
                                   rel_clip=None, abs_clip=None)

    @pytest.fixture()
    def em(self):
        """Setup"""
        xyz = torch.tensor([[10., 50., 100.]])
        return emitter.CoordinateOnlyEmitter(xyz, xy_unit='nm')

    def test_forward(self, rend, em):
        histogram = rend.forward(em)
        assert histogram.size() == torch.Size([10, 10])

    @pytest.mark.plot
    def test_plot_frame_render_visual(self, rend, em):
        PlotFrameCoord(torch.zeros((101, 101)), em.xyz_nm).plot()
        plt.show()

        rend.render(em)
        plt.show()
        
    
class TestRenderer3D:

    @pytest.fixture()
    def rend(self):
        return renderer.Renderer3D(plot_axis = (0,1,2), xextent=(0., 100.), yextent=(0., 100.), zextent=(-100., 100.), px_size=10., sigma_blur=10.,
                                   rel_clip=None, abs_clip=None)

    @pytest.fixture()
    def em(self):
        """Setup"""
        xyz = torch.tensor([[10., 50., 100.]])
        return emitter.CoordinateOnlyEmitter(xyz, xy_unit='nm')

    def test_forward(self, rend, em):
        histogram = rend.forward(em)
        assert histogram.size() == torch.Size([10, 10, 3])

    @pytest.mark.plot
    def test_plot_frame_render_visual(self, rend, em):
        PlotFrameCoord(torch.zeros((101, 101)), em.xyz_nm).plot()
        plt.show()

        rend.render(em)
        plt.show()

