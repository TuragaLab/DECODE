import numpy as np
import pytest
import torch

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


class TestNoPostProcessing(TestPostProcessingAbstract):
    @pytest.fixture()
    def post(self):
        return post_processing.NoPostProcessing()


class TestLookUpPostProcessing(TestPostProcessingAbstract):
    @pytest.fixture()
    def post(self):
        return post_processing.LookUpPostProcessing(raw_th=0.1, xy_unit="px")

    @pytest.fixture()
    def pseudo_out_no_sigma(self):
        """Pseudo model output without sigma prediction"""

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

    @pytest.fixture()
    def pseudo_out(self):
        """Pseudo model output with sigma prediction"""

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

        active_px = post._filter(detection)

        assert (active_px == torch.tensor([[1, 0], [1, 0]]).unsqueeze(0).bool()).all()

        return active_px

    def test_lookup(self, post):

        active_px = self.test_filter(post)  # get the return value of the previous test

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

        post.photxyz_sigma_mapping = (
            None  # because this test is without sigma (this is non-default)
        )

        emitter_out = post.forward(pseudo_out_no_sigma)

        assert isinstance(
            emitter_out, emitter.EmitterSet
        ), "Output should be an emitter."
        assert (emitter_out.frame_ix == 0).all()
        assert (emitter_out.phot.unique() == torch.tensor([1.0, 3.0])).all()

    def test_forward(self, post, pseudo_out):
        emitter_out = post.forward(pseudo_out)

        assert isinstance(
            emitter_out, emitter.EmitterSet
        ), "Output should be an emitter."
        assert (emitter_out.frame_ix == 0).all()
        assert (emitter_out.phot.unique() == torch.tensor([1.0, 3.0])).all()

        assert not torch.isnan(
            emitter_out.xyz_sig
        ).any(), "Sigma values for xyz should not be nan."
        assert not torch.isnan(
            emitter_out.phot_sig
        ).any(), "Sigma values for phot should not be nan."
        assert emitter_out.bg_sig is None

        assert test_utils.tens_almeq(
            emitter_out.xyz_sig,
            torch.tensor([[20.0, 30.0, 40.0], [2 / 0.6, 3 / 0.6, 4 / 0.6]]),
        )

        assert test_utils.tens_almeq(
            emitter_out.phot_sig, torch.tensor([10.0, 1 / 0.6])
        )


class TestSpatialIntegration(TestLookUpPostProcessing):
    @pytest.fixture()
    def post(self):
        return post_processing.SpatialIntegration(raw_th=0.1, xy_unit="px")

    def test_forward_no_sigma(self, post, pseudo_out_no_sigma):
        # because this test is without sigma (this is non-default)
        post.photxyz_sigma_mapping = None

        emitter_out = post.forward(pseudo_out_no_sigma)

        assert isinstance(
            emitter_out, emitter.EmitterSet
        ), "Output should be an emitter."
        assert len(emitter_out) == 1
        assert emitter_out.frame_ix == 0
        assert emitter_out.phot == pytest.approx(3.0)
        assert emitter_out.prob == pytest.approx(0.75)

    def test_forward(self, post, pseudo_out):
        emitter_out = post.forward(pseudo_out)

        assert isinstance(emitter_out, emitter.EmitterSet)
        assert len(emitter_out) == 1

        assert not torch.isnan(
            emitter_out.xyz_sig
        ).any(), "Sigma values for xyz should not be nan."
        assert not torch.isnan(
            emitter_out.phot_sig
        ).any(), "Sigma values for phot should not be nan."
        assert emitter_out.bg_sig is None

        np.testing.assert_allclose(
            emitter_out.xyz_sig, torch.tensor([[2 / 0.6, 3 / 0.6, 4 / 0.6]])
        )
        np.testing.assert_allclose(emitter_out.phot_sig, torch.tensor([1 / 0.6]))

    @pytest.mark.parametrize(
        "aggr,expct",
        [
            ("sum", ([0.0, 1.000001], [0.0, 0.501])),
            ("norm_sum", ([0.0, 1.0], [0.0, 0.501])),
        ],
    )
    def test_nms(self, post, aggr, expct):
        """Setup, Run, Assert"""
        post.p_aggregation = post.set_p_aggregation(aggr)

        p = torch.zeros((2, 32, 32))
        p[0, 4:6, 4] = 0.5
        p[0, 4, 4] = 0.5
        p[0, 5, 4] = 0.500001
        p[1, 6, 4] = 0.25
        p[1, 7, 4] = 0.251

        p_out = post._nms(p, post.p_aggregation, 0.2, 0.6)

        np.testing.assert_allclose(p_out[0, 4:6, 4], torch.tensor(expct[0]), atol=1e-5)
        np.testing.assert_allclose(p_out[1, 6:8, 4], torch.tensor(expct[1]), atol=1e-5)


# ToDo: This test should be rewritten if this class is in use
class TestConsistentPostProcessing(TestPostProcessingAbstract):
    @pytest.fixture()
    def post(self):
        return post_processing.ConsistencyPostprocessing(raw_th=0.1, em_th=0.5, xy_unit="px",
                                                         img_shape=(32, 32), ax_th=None, lat_th=0.5,
                                                         match_dims=2)

    def test_excpt(self, post):
        with pytest.raises(IndexError):
            post.forward(torch.rand((1, 2, 32, 32)))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available on this machine."
    )
    def test_forward_cuda(self, post):
        p = torch.zeros((2, 1, 32, 32)).cuda()
        out = torch.zeros((2, 5, 32, 32)).cuda()
        p[1, 0, 2, 4] = 0.6
        p[1, 0, 2, 6] = 0.6
        p[0, 0, 0, 0] = 0.3
        p[0, 0, 0, 1] = 0.4

        out[0, 2, 0, 0] = 0.3
        out[0, 2, 0, 1] = 0.5
        out[1, 2, 2, 4] = 1.0
        out[1, 2, 2, 6] = 1.2

        _ = post.forward(torch.cat((p, out), 1))

    @pytest.mark.skip(
        reason="Implementation was not robust and was removed. "
        "Test can be used when new implementation is there."
    )
    def test_forward_multi_worker(self, post):

        p = torch.zeros((2, 1, 32, 32))
        out = torch.zeros((2, 5, 32, 32))
        p[1, 0, 2, 4] = 0.6
        p[1, 0, 2, 6] = 0.6
        p[0, 0, 0, 0] = 0.3
        p[0, 0, 0, 1] = 0.4

        out[0, 2, 0, 0] = 0.3
        out[0, 2, 0, 1] = 0.5
        out[1, 2, 2, 4] = 1.0
        out[1, 2, 2, 6] = 1.2

        out[:, 4] = torch.rand_like(out[:, 4])

        post.num_workers = 0
        em0 = post.forward(torch.cat((p, out), 1))

        post.num_workers = 4
        em1 = post.forward(torch.cat((p, out), 1))

        for i in range(len(em0)):
            assert em0[i] == em1[i]

    def test_forward(self, post):
        """Easy case, i.e. isolated active pixels."""

        p = torch.zeros((2, 1, 32, 32))
        out = torch.zeros((2, 5, 32, 32))
        p[0, 0, 0, 0] = 0.3
        p[0, 0, 0, 2] = 0.4
        p[1, 0, 2, 4] = 0.6
        p[1, 0, 2, 6] = 0.6

        out[0, 2, 0, 0] = 0.3
        out[0, 2, 0, 2] = 0.5
        out[1, 2, 2, 4] = 1.0
        out[1, 2, 2, 6] = 1.2

        p_out, feat_out = post._forward_raw_impl(p, out)
        em_out = post.forward(torch.cat((p, out), 1))

        assert test_utils.tens_almeq(p, p_out)
        assert test_utils.tens_almeq(out, feat_out)

        assert isinstance(em_out, emitter.EmitterSet)
        assert len(em_out) == 2
        assert (em_out.prob >= post.em_th).all()

    def test_forward_hard_cases(self, post):
        """Non-isolated emitters."""

        p = torch.zeros((3, 1, 32, 32))
        out = torch.zeros((3, 5, 32, 32))
        p[0, 0, 0, 0] = 0.7  # isolated
        p[0, 0, 0, 2] = 0.7  # isolated
        p[1, 0, 2, 4] = 0.6  # should be merged
        p[1, 0, 2, 5] = 0.6  # should be merged
        p[2, 0, 4, 4] = 0.7  # should not be merged
        p[2, 0, 4, 5] = 0.7  # should not be merged

        out[1, 1, 2, 4] = 20.0  # should be merged
        out[1, 1, 2, 5] = 20.2  # %
        out[2, 2, 4, 4] = 49.0  # should not be merged
        out[2, 2, 4, 5] = 49.51  # %

        em_out = post.forward(torch.cat((p, out), 1))

        # First frame
        assert len(em_out.iframe[0]) == 2
        assert (em_out.iframe[0].prob == 0.7).all()

        # Second frame
        assert len(em_out.iframe[1]) == 1
        assert em_out.iframe[1].prob.item() > 0.6

        # Third frame
        assert len(em_out.iframe[2]) == 2
        assert (em_out.iframe[2].prob == 0.7).all()

    @pytest.mark.parametrize(
        "x,expct",
        [
            (torch.ones((2, 6, 32, 32)), True),
            (torch.zeros((2, 6, 32, 32)), False),
            (
                torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
                .unsqueeze(0)
                .unsqueeze(0),
                False,
            ),
        ],
    )
    def test_filter(self, post, x, expct):
        post.skip_th = 0.2
        assert post.skip_if(x) is expct
