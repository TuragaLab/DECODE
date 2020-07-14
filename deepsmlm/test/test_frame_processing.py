import torch
import pytest

from deepsmlm.neuralfitter import frame_processing
from deepsmlm.generic import test_utils


class TestMirror:

    @pytest.fixture()
    def proc(self):
        return frame_processing.Mirror2D(dims=(-2, -1))

    def test_forward(self, proc):

        """Setup"""
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        x_tar = torch.tensor([[6., 5., 4.], [3., 2., 1.]])

        """Run"""
        x_out = proc.forward(x)

        """Assert"""
        assert test_utils.tens_almeq(x_out, x_tar)
