import torch
import pytest

from decode.neuralfitter import frame_processing
from decode.generic import test_utils


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

    def test_forward_noop(self, proc):
        proc.px_fold = 1

        x = torch.rand(2, 32, 34)
        out = proc.forward(x)

        assert out.size() == x.size()
        assert (out == x).all()


class TestAutoPad(TestAutoCenterCrop):

    @pytest.fixture()
    def proc(self):
        return frame_processing.AutoPad(4)

    @pytest.mark.parametrize("actual_size,target_size", [((5, 6), (8, 8)),
                                                         ((5, 8), (8, 8)),
                                                         ((12, 4), (12, 4)),
                                                         ((13, 25), (16, 28)),
                                                         ((3, 8), (4, 8))])
    def test_forward(self, actual_size, target_size, proc):

        x = torch.rand((2, *actual_size))

        """Forward"""
        if target_size == 'err':
            with pytest.raises(ValueError):
                proc.forward(x)

        else:
            x_out = proc.forward(x)
            assert x_out[0].size() == torch.Size(target_size)



@pytest.mark.parametrize("x,size_out", [(torch.rand(3, 62, 68), torch.Size([3, 59, 65])), # size out of mock pipeline
                                        (torch.rand(2, 3, 59, 68), torch.Size([2, 3, 56, 65]))])
def test_get_frame_extent(x, size_out):

    """Setup"""
    class Pipeline:
        def forward(self, x):
            return x[..., :-3, :-3]

    pipeline = Pipeline()

    """Run"""
    frame_size = frame_processing.get_frame_extent(x.size(), pipeline.forward)

    """Assert"""
    assert frame_size == size_out
