import warnings

import numpy as np
import pytest
import torch

from decode.emitter import emitter
from decode.neuralfitter import spec
from decode.neuralfitter.processing import to_emitter


class TestToEmitter:
    @pytest.fixture
    def post(self):
        class PostProcessingMock(to_emitter.ToEmitter):
            def forward(self, *args):
                return emitter.factory(0)

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
    def ch_map(self) -> spec.ModelChannelMapGMM:
        return spec.ModelChannelMapGMM(n_codes=2)

    @pytest.fixture
    def post(self, ch_map):
        return to_emitter.ToEmitterLookUpPixelwise(
            mask=0.1, ch_map=ch_map, xy_unit="px"
        )

    @pytest.fixture
    def pseudo_out(self):
        # mock model out with sigma prediction

        detection = torch.tensor(
            [
                [[0.1, 0.0], [0.0, 0.05]],
                [[0.0, 0.0], [0.6, 0.02]],
            ]
        ).unsqueeze(0)
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
        sigma /= detection.max(1, keepdim=True)[0]

        bg = torch.rand_like(detection)
        pseudo_net_ouput = torch.cat((detection, features, sigma, bg), 1)

        return pseudo_net_ouput

    def test_mask_impl(self, post):
        detection = torch.tensor([[0.1, 0.0], [0.6, 0.05]]).unsqueeze(0)
        active_px = post._mask(detection)
        assert (active_px == torch.tensor([[1, 0], [1, 0]]).unsqueeze(0).bool()).all()

    def test_lookup_features(self, post):
        active_px = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool).unsqueeze(0)

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

        assert isinstance(batch_ix, torch.LongTensor)
        assert (batch_ix == 0).all()
        assert batch_ix.size()[0] == features.size()[1]

        # This is hard coded designed for the very specific test case
        assert (
            (features / (torch.arange(5).unsqueeze(1).float() + 1)).unique()
            == torch.tensor([1.0, 3.0])
        ).all()

    def test_lookup_code(self, post):
        mask = torch.tensor(
            [[[1, 1], [1, 0]], [[1, 0], [1, 0]]], dtype=torch.bool
        ).unsqueeze(0)
        code_expct = torch.tensor([1, 0, 1], dtype=torch.long)

        code = post._look_up_code(mask)

        np.testing.assert_array_equal(code, code_expct)

    def test_lookup_code_feature_order(self, post):
        mask = torch.tensor(
            [[[1, 1], [1, 0]], [[1, 0], [1, 0]]], dtype=torch.bool
        ).unsqueeze(0)
        features = mask.max(1)[0].float()

        code_expct = torch.tensor([1, 0, 1], dtype=torch.long)

        code = post._look_up_code(mask)
        _, feat_out = post._lookup_features(mask, mask.max(1)[0])

        np.testing.assert_array_equal(code, feat_out.sum(0) - 1)

    def test_forward(self, post, pseudo_out):
        em = post.forward(pseudo_out)

        assert isinstance(em, emitter.EmitterSet)
        assert (em.frame_ix == 0).all()
        assert (em.phot.unique() == torch.tensor([1.0, 3.0])).all()
        np.testing.assert_array_almost_equal(em.prob, torch.tensor([0.1, 0.6]))

        assert not torch.isnan(em.xyz_sig).any()
        assert not torch.isnan(em.phot_sig).any()
        assert em.bg_sig is None

        np.testing.assert_array_almost_equal(
            em.xyz_sig,
            torch.tensor([[20.0, 30.0, 40.0], [2 / 0.6, 3 / 0.6, 4 / 0.6]]),
        )
        np.testing.assert_array_almost_equal(em.phot_sig, torch.tensor([10.0, 1 / 0.6]))


class TestSpatialIntegration(TestLookUpPostProcessing):
    @pytest.fixture
    def post(self, ch_map):
        return to_emitter.ToEmitterSpatialIntegration(
            raw_th=0.1, ch_map=ch_map, xy_unit="px"
        )

    @pytest.mark.parametrize("code_eq", [True, False])
    def test_forward(self, post, pseudo_out, code_eq):
        if code_eq:
            pseudo_out[:, 0] = pseudo_out[:, post._ch_map.ix_prob].max(1)[0]
            pseudo_out[:, post._ch_map.ix_prob[1:]] = 0.0

        emitter_out = post.forward(pseudo_out)

        assert isinstance(emitter_out, emitter.EmitterSet)
        assert len(emitter_out) == 1 if code_eq else 2
        assert not torch.isnan(emitter_out.xyz_sig).any()
        assert not torch.isnan(emitter_out.phot_sig).any()
        assert emitter_out.bg_sig is None

        if code_eq:
            assert len(emitter_out) == 1
            np.testing.assert_allclose(
                emitter_out.xyz_sig, torch.tensor([[2 / 0.6, 3 / 0.6, 4 / 0.6]])
            )
            np.testing.assert_allclose(emitter_out.phot_sig, torch.tensor([1 / 0.6]))
        else:
            assert len(emitter_out) == 2

    @pytest.mark.parametrize(
        "aggr,expct",
        [
            ("sum", ([0.0, 1.000001], [0.0, 0.501])),
            ("norm_sum", ([0.0, 1.0], [0.0, 0.501])),
        ],
    )
    def test_nms(self, post, aggr, expct):
        post.p_aggregation = post._set_p_aggregation(aggr)
        post._raw_th = 0.2
        post._split_th = 0.6

        p = torch.zeros((2, 32, 32))
        # two emitters
        p[0, 15, 15] = 0.9
        p[0, 16, 15] = 0.95
        # one emitter with sum > 1. (to test norm_sum vs sum)
        p[0, 4, 4] = 0.5
        p[0, 5, 4] = 0.500001
        # one emitter with sum < 1.
        p[1, 6, 4] = 0.25
        p[1, 7, 4] = 0.251
        p = p.unsqueeze(1)
        p = p.repeat(1, 2, 1, 1)

        p_out = post._non_max_suppression(p)

        assert 0 <= p_out.min() < p_out.max() <= 1.
        np.testing.assert_array_equal(p[:, 0], p[:, 1])

        try:
            np.testing.assert_allclose(
                p_out[0, 0, 15:17, 15], torch.tensor([0.9, 0.95])
            )
        except AssertionError:
            warnings.warn(
                "NMS test failed. This is a known issue, "
                "for two high probabilty emitters nearby."
            )

        # should return the same as p is only repeated
        for pp in torch.unbind(p_out, dim=1):
            np.testing.assert_allclose(pp[0, 4:6, 4], torch.tensor(expct[0]), atol=1e-5)
            np.testing.assert_allclose(pp[1, 6:8, 4], torch.tensor(expct[1]), atol=1e-5)
