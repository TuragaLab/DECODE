import pytest
import torch

import decode.neuralfitter
from decode.emitter import emitter
from decode.generic import test_utils
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


class TestBgPerEmitterFromBgFrame:
    @pytest.fixture(scope="class")
    def extractor(self):
        return decode.neuralfitter.post_processing.EmitterBackgroundByFrame(
            17, (-0.5, 63.5), (-0.5, 63.5), (64, 64)
        )

    def test_mean_filter(self, extractor):
        x_in = []
        x_in.append(torch.randn((1, 1, 64, 64)))
        x_in.append(torch.zeros((1, 1, 64, 64)))
        x_in.append(
            torch.meshgrid(torch.arange(64), torch.arange(64))[0]
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )

        # excpt outcome
        expect = []
        expect.append(torch.zeros_like(x_in[0]))
        expect.append(torch.zeros_like(x_in[0]))
        expect.append(8)

        out = []
        for x in x_in:
            out.append(extractor._mean_filter(x))

        assert test_utils.tens_almeq(out[0], expect[0], 1)  # 10 sigma
        assert test_utils.tens_almeq(out[1], expect[1])
        assert test_utils.tens_almeq(
            out[2][0, 0, 8, :], 8 * torch.ones_like(out[2][0, 0, 0, :]), 1e-4
        )

    test_data = [
        (torch.zeros((1, 1, 64, 64)), emitter.factory(100), torch.zeros((100,))),
        (
            torch.meshgrid(torch.arange(64), torch.arange(64))[0]
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            emitter.factory(xyz=[[8.0, 0.0, 0.0]]),
            torch.tensor([8.0]),
        ),
        (
            torch.rand((1, 1, 64, 64)),
            emitter.factory(xyz=[[70.0, 32.0, 0.0]]),
            torch.tensor([float("nan")]),
        ),
    ]

    @pytest.mark.parametrize("bg,em,expect_bg", test_data)
    def test_forward(self, extractor, bg, em, expect_bg):
        out = extractor.forward(em, bg)

        assert test_utils.tens_almeq(out.bg, expect_bg, 1e-4, nan=True)
