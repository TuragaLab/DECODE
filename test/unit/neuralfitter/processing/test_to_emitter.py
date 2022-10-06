import numpy as np
import pytest
import torch

from decode.emitter import emitter
from decode.neuralfitter.processing import to_emitter


class TestToEmitter:
    @pytest.fixture
    def post(self):
        class PostProcessingMock(to_emitter.ToEmitter):
            def forward(self, *args):
                return emitter.EmptyEmitterSet()

        return PostProcessingMock()

    def test_forward(self, post):
        e = post.forward()
        assert isinstance(e, emitter.EmitterSet)


class TestNoPostProcessing(TestToEmitter):
    @pytest.fixture
    def post(self):
        return to_emitter.ToEmitterEmpty()


class TestLookUpPostProcessing(TestToEmitter):
    @pytest.fixture
    def post(self):
        return to_emitter.ToEmitterLookUpPixelwise(mask=0.1, xy_unit="px")

    @pytest.fixture
    def pseudo_out_no_sigma(self):
        detection = torch.tensor([[0.1, 0.0], [0.6, 0.05]]).unsqueeze(0).unsqueeze(0)
        features = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 5, 1, 1)
        )
        features = features * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(
            0
        ).unsqueeze(-1).unsqueeze(-1)

        pseudo_net_ouput = torch.cat((detection, features), 1)

        return pseudo_net_ouput

    @pytest.fixture
    def pseudo_out(self):
        """Mock model out with sigma prediction"""

        detection = torch.tensor([[0.1, 0.0], [0.6, 0.05]]).unsqueeze(0).unsqueeze(0)
        features = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 4, 1, 1)
        )
        features = features * torch.tensor([1.0, 2.0, 3.0, 4.0]).unsqueeze(0).unsqueeze(
            -1
        ).unsqueeze(-1)
        sigma = torch.ones((1, 4, 2, 2))
        sigma *= torch.arange(1, 5).view(1, -1, 1, 1)
        sigma /= detection

        pseudo_net_ouput = torch.cat(
            (detection, features, sigma, torch.rand_like(detection)), 1
        )

        return pseudo_net_ouput

    def test_filter(self, post):
        detection = torch.tensor([[0.1, 0.0], [0.6, 0.05]]).unsqueeze(0)
        active_px = post._mask(detection)
        assert (active_px == torch.tensor([[1, 0], [1, 0]]).unsqueeze(0).bool()).all()

    def test_lookup(self, post):
        active_px = post._mask(torch.tensor([[0.1, 0.0], [0.6, 0.05]]).unsqueeze(0))

        features = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 5, 1, 1)
        )
        features = features * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(
            0
        ).unsqueeze(-1).unsqueeze(-1)

        batch_ix, features = post._lookup_features(features, active_px)

        assert isinstance(
            batch_ix, torch.LongTensor
        ), "Batch ix should be integer type."
        assert (batch_ix == 0).all()
        assert batch_ix.size()[0] == features.size()[1]

        # This is hard coded designed for the very specific test case
        assert (
                (features / (torch.arange(5).unsqueeze(1).float() + 1)).unique()
                == torch.tensor([1.0, 3.0])
        ).all()

    def test_forward_no_sigma(self, post, pseudo_out_no_sigma):
        post._photxyz_sigma_mapping = (
            None  # because this test is without sigma (this is non-default)
        )

        emitter_out = post.forward(pseudo_out_no_sigma)

        assert isinstance(emitter_out, emitter.EmitterSet)
        assert (emitter_out.frame_ix == 0).all()
        assert (emitter_out.phot.unique() == torch.tensor([1.0, 3.0])).all()

    def test_forward(self, post, pseudo_out):
        emitter_out = post.forward(pseudo_out)

        assert isinstance(emitter_out, emitter.EmitterSet)
        assert (emitter_out.frame_ix == 0).all()
        assert (emitter_out.phot.unique() == torch.tensor([1.0, 3.0])).all()

        assert not torch.isnan(
            emitter_out.xyz_sig
        ).any(), "Sigma values for xyz should not be nan."
        assert not torch.isnan(
            emitter_out.phot_sig
        ).any(), "Sigma values for phot should not be nan."
        assert emitter_out.bg_sig is None

        np.testing.assert_array_almost_equal(
            emitter_out.xyz_sig,
            torch.tensor([[20.0, 30.0, 40.0], [2 / 0.6, 3 / 0.6, 4 / 0.6]]),
        )
        np.testing.assert_array_almost_equal(
            emitter_out.phot_sig, torch.tensor([10.0, 1 / 0.6])
        )
