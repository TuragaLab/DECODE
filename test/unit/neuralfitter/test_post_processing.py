import pytest
import torch

from decode.emitter import emitter
from decode.neuralfitter import post_processing


class TestPostProcessingAbstract:
    @pytest.fixture()
    def post(self):
        class PostProcessingMock(post_processing.PostProcessing):
            def forward(self, *args):
                return emitter.EmptyEmitterSet()

        return PostProcessingMock(xy_unit=None, px_size=None)

    def test_forward(self, post):
        e = post.forward()
        assert isinstance(e, emitter.EmitterSet)

    def test_skip_if(self, post):
        assert not post.skip_if(torch.rand((1, 3, 32, 32)))


@pytest.mark.skip(reason="deprecated impl")
class TestConsistentPostProcessing(TestPostProcessingAbstract):
    pass


