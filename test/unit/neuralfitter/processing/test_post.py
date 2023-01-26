import pytest
import torch

from decode.emitter import emitter
from decode.neuralfitter import spec
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
        ch_map = spec.ModelChannelMapGMM(3)
        scaler = scale_transform.ScalerModelChannel.from_ch_spec(
            ch_map=ch_map,
            phot=10000.,
            z=800.,
            bg=100.,
            sigma_factor=1.,
            sigma_eps=0.,
        )
        coord = coord_transform.Offset2Coordinate((-0.5, 31.5), (-0.5, 31.5), (32, 32))
        frame2em = to_emitter.ToEmitterLookUpPixelwise(.6, ch_map=ch_map, xy_unit="px")

        return post_processing.PostProcessingGaussianMixture(
            scaler=scaler,
            coord_convert=coord,
            frame_to_emitter=frame2em,
            ch_map=ch_map,
        )

    def test_forward(self, post):
        x = torch.rand(2, 14, 32, 32)
        if x.size(1) != post._ch_map.n:
            raise ValueError("Wrong number of channels")

        # set most of the probs to zero
        x[:, post._ch_map.ix_prob][x[:, post._ch_map.ix_prob] < 0.98] = 0

        # set x/y offset vectors to default offset range -0.5 to 0.5
        x[:, post._ch_map.ix_xyz[:-1]] -= 0.5

        em = post.forward(x)

        assert isinstance(em, emitter.EmitterSet)
        assert em.xyz[:, :2].min() >= -0.5
        assert em.xyz[:, :2].max() < 31.5
        assert (em.xyz[:, 2] > 100).any()
        assert em.xyz[:, 2].max() < 800


@pytest.mark.skip(reason="deprecated impl")
class TestConsistentPostProcessing(TestPostProcessingAbstract):
    pass
