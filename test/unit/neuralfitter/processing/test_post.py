import pytest
import torch

from decode.emitter import emitter
from decode.neuralfitter.processing import post as post_processing
from decode.neuralfitter import scale_transform, coord_transform
from decode.neuralfitter.processing import to_emitter


class TestPostProcessingAbstract:
    @pytest.fixture()
    def post(self):
        class PostProcessingMock(post_processing.PostProcessing):
            def forward(self, *args):
                return emitter.EmptyEmitterSet()

        return PostProcessingMock()

    def test_forward(self, post):
        e = post.forward(torch.rand(1, 3, 32, 32))
        assert isinstance(e, emitter.EmitterSet)

    def test_skip_if(self, post):
        assert not post.skip_if(torch.rand((1, 3, 32, 32)))


class TestPostProcessingGaussianMixture(TestPostProcessingAbstract):
    @pytest.fixture
    def post(self):
        scaler = scale_transform.ScalerModelOutput(1., 2., 3.)
        coord = coord_transform.Offset2Coordinate((-0.5, 31.5), (-0.5, 31.5), (32, 32))
        frame2em = to_emitter.ToEmitterLookUpPixelwise(.6, xy_unit="px")

        return post_processing.PostProcessingGaussianMixture(scaler, coord, frame2em)

    def test_forward(self, post):
        x = torch.rand(2, 10, 32, 32)
        x[x > 0.01] = 0

        em = post.forward(x)
        assert isinstance(em, emitter.EmitterSet)


@pytest.mark.skip(reason="deprecated impl")
class TestConsistentPostProcessing(TestPostProcessingAbstract):
    pass
