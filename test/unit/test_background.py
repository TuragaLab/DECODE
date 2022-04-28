from abc import ABC

import pytest
import torch

import decode.simulation.background as background
from decode.generic import emitter, test_utils


class BackgroundAbstractTest(ABC):

    @pytest.fixture()
    def bgf(self):
        raise NotImplementedError

    @pytest.fixture()
    def rframe(self):
        """
        Just a random frame batch.

        Returns:
            rframe (torch.Tensor): random frame batch

        """
        return torch.rand((2, 3, 64, 64))

    def test_sanity(self, bgf):
        with pytest.raises(ValueError) as e_info:
            bgf.__init__(forward_return='asjdfki')

    def test_sample(self, bgf):
        """Tests sampler"""

        out = bgf.sample(size=(32, 32), device=torch.device('cpu'))

        assert out.size() == torch.Size((32, 32))

    def test_sample_like(self, bgf):
        """Tests sampler"""

        x = torch.rand((32, 32)).double()

        out = bgf.sample_like(x)

        assert x.size() == out.size()
        assert x.device == out.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Makes only sense if we have another device to test on.")
    def test_sample_like_cuda(self, bgf):
        x = torch.rand((2, 32, 32)).cuda()

        out = bgf.sample_like(x)

        assert x.size() == out.size()
        assert x.device == out.device

    def test_shape(self, bgf, rframe):
        """
        Tests shape equality

        Args:
            bgf: fixture as above
            rframe: fixture as above

        """
        rframe_bg, bg_term = bgf.forward(rframe)

        assert rframe_bg.size() == rframe.size(), "Wrong shape after background."
        assert bg_term.dim() <= rframe_bg.dim(), "Background term must be of less or equal dimension than input frame."

    def test_additivity(self, bgf, rframe):
        """
        Tests whether the backgorund is additive

        Args:
            bgf: fixture as above
            rframe: fixture as above

        """
        rframe_bg, bg_term = bgf.forward(rframe)
        assert (rframe_bg >= rframe).all(), "Background is not strictly unsigned additive."

        _ = rframe + bg_term  # test whether one can add background to the rframe so that dimension etc. are correct


class TestUniformBackground(BackgroundAbstractTest):

    @pytest.fixture()
    def bgf(self):
        return background.UniformBackground((0., 100.), forward_return='tuple')

    def test_sample(self, bgf):
        super().test_sample(bgf)

        out = bgf.sample((5, 32, 32))

        assert len(out.unique()) == 5, "Should have as many unique values as we have batch size."

        for out_c in out:
            assert len(out_c.unique()) == 1, "Background not spacially constant"

        assert ((out >= 0) * (out <= 100)).all(), "Wrong output values."


class TestBgPerEmitterFromBgFrame:

    @pytest.fixture(scope='class')
    def extractor(self):
        return background.BgPerEmitterFromBgFrame(17, (-0.5, 63.5), (-0.5, 63.5), (64, 64))

    def test_mean_filter(self, extractor):
        """
        Args:
            extractor: fixture as above

        """
        """Some hard coded setups"""
        x_in = []
        x_in.append(torch.randn((1, 1, 64, 64)))
        x_in.append(torch.zeros((1, 1, 64, 64)))
        x_in.append(torch.meshgrid(torch.arange(64), torch.arange(64))[0].unsqueeze(0).unsqueeze(0).float())

        # excpt outcome
        expect = []
        expect.append(torch.zeros_like(x_in[0]))
        expect.append(torch.zeros_like(x_in[0]))
        expect.append(8)

        """Run"""
        out = []
        for x in x_in:
            out.append(extractor._mean_filter(x))

        """Assertions"""
        assert test_utils.tens_almeq(out[0], expect[0], 1)  # 10 sigma
        assert test_utils.tens_almeq(out[1], expect[1])
        assert test_utils.tens_almeq(out[2][0, 0, 8, :], 8 * torch.ones_like(out[2][0, 0, 0, :]), 1e-4)

    test_data = [
        (torch.zeros((1, 1, 64, 64)), emitter.RandomEmitterSet(100), torch.zeros((100,))),
        (torch.meshgrid(torch.arange(64), torch.arange(64))[0].unsqueeze(0).unsqueeze(0).float(),
         emitter.CoordinateOnlyEmitter(torch.tensor([[8., 0., 0.]])),
         torch.tensor([8.])),
        (torch.rand((1, 1, 64, 64)), emitter.CoordinateOnlyEmitter(torch.tensor([[70., 32., 0.]])),
         torch.tensor([float('nan')]))
    ]

    @pytest.mark.parametrize("bg,em,expect_bg", test_data)
    def test_forward(self, extractor, bg, em, expect_bg):
        """Run"""
        out = extractor.forward(em, bg)

        """Assertions"""
        assert test_utils.tens_almeq(out.bg, expect_bg, 1e-4, nan=True)
