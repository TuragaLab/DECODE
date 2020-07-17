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


class TestAutoCenterCrop:

    @pytest.fixture()
    def proc(self):
        return frame_processing.AutoCenterCrop(4)

    @pytest.mark.parametrize("actual_size,target_size", [((5, 6), (4, 4)),
                                                         ((5, 8), (4, 8)),
                                                         ((12, 4), (12, 4)),
                                                         ((13, 25), (12, 24)),
                                                         ((3, 8), 'err')])
    def test_forward(self, actual_size, target_size, proc):

        """Setup"""
        x = torch.rand((2, *actual_size))

        """Forward"""
        if target_size == 'err':
            with pytest.raises(ValueError):
                proc.forward(x)

        else:
            x_out = proc.forward(x)
            assert x_out[0].size() == torch.Size(target_size)