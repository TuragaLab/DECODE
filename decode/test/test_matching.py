import warnings

import pytest
import torch

import decode.evaluation
import decode.evaluation.match_emittersets as match_em
import decode.generic
from decode import CoordinateOnlyEmitter


class TestMatcherABC:
    """
    Defines tests that should succeed on all implementations of a matching algorithm. All test classes should
    therefore inherit this test.
    """

    @pytest.fixture()
    def matcher(self):
        class MockMatch(decode.evaluation.match_emittersets.EmitterMatcher):
            def forward(self, output, target):
                """Randomly assign tp and tp_match"""
                ix_tp = torch.randint(2, size=(len(output),)).bool()  # assign output randomly as tp / fp
                ix_tp_match = torch.zeros(len(target)).bool()  # init ix for gt

                n_tp = (ix_tp == 1).nonzero().numel()
                ix_tp_match[:n_tp] = 1  # set first emitters of gt to be the matching emitters

                tp = output[ix_tp]
                tp_match = target[ix_tp_match]

                fp = output[~ix_tp]
                fn = target[~ix_tp_match]

                return self._return_match(tp=tp, fp=fp, fn=fn, tp_match=tp_match)

        return MockMatch()

    @pytest.fixture()
    def can_em_out(self):  # candidate emitter output
        em = decode.generic.emitter.RandomEmitterSet(1000, xy_unit='nm')
        em.frame_ix = torch.randint_like(em.frame_ix, 100)

        return em

    @pytest.fixture()
    def can_em_tar(self, can_em_out):  # candidate emitter target
        em = decode.generic.emitter.RandomEmitterSet(len(can_em_out), xy_unit='nm')
        em.frame_ix = torch.randint_like(em.frame_ix, 50)

        return em

    def test_split_sanity(self, matcher, can_em_out, can_em_tar):
        """
        Tests the return sanity, i.e. number of elements etc.
        """

        """Run"""
        tp, fp, fn, tp_match = matcher.forward(can_em_out, can_em_tar)

        """Asserts"""
        assert len(tp) == len(tp_match), "Inconsistent number of emitters for true positives and matching ground " \
                                         "truths."
        assert len(tp) + len(fp) == len(can_em_out), "Inconsistent split in true positives and false positives."
        assert len(can_em_tar) - len(tp_match) == len(fn), "Inconsistent split."

        if not (tp.id == tp_match.id).all():
            warnings.warn("Match implementation does not match identities. Probably not on purpose?")


class TestGreedyMatching(TestMatcherABC):

    @pytest.fixture()
    def matcher(self):
        return match_em.GreedyHungarianMatching(match_dims=2)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_init(self, dim):
        """Tests the safety checks"""

        """Expected to raise expections:"""
        with pytest.raises((ValueError, TypeError)):
            match_em.GreedyHungarianMatching()  # match_dims missing

        with pytest.warns(UserWarning):
            match_em.GreedyHungarianMatching(match_dims=dim, dist_lat=1., dist_ax=1., dist_vol=1.)  # unlikely comb.

        with pytest.warns(UserWarning):
            match_em.GreedyHungarianMatching(match_dims=dim)  # unlikely comb.

        with pytest.raises(ValueError):
            match_em.GreedyHungarianMatching(match_dims=1)

        with pytest.raises(ValueError):
            match_em.GreedyHungarianMatching(match_dims=4)

    @staticmethod
    def assert_dists(r_out, r_tar, dist_lat, dist_ax, dist_vol, filter_mask):

        assert isinstance(filter_mask, torch.BoolTensor)

        if dist_lat is not None:
            dist_mat = torch.cdist(r_out[:, :, :2], r_tar[:, :, :2], p=2).sqrt()
            assert (dist_mat[filter_mask] <= dist_lat).all()

        if dist_ax is not None:
            dist_mat = torch.cdist(r_out[:, :, [2]], r_tar[:, :, [2]], p=2).sqrt()
            assert (dist_mat[filter_mask] <= dist_ax).all()

        if dist_vol is not None:
            dist_mat = torch.cdist(r_out, r_tar, p=2).sqrt()
            assert (dist_mat[filter_mask] <= dist_vol).all()

    def test_filter_kernel_hand(self):

        """Setup"""
        matcher = match_em.GreedyHungarianMatching(match_dims=2, dist_lat=2., dist_ax=None, dist_vol=None)

        """Run"""
        filter = matcher.filter(torch.zeros((4, 3)),
                                torch.tensor([[1.9, 0., 0.], [2.1, 0., 0.], [0., 0., -5000.], [1.5, 1.5, 0.]]))

        """Assert"""
        assert filter[:, 0].all()
        assert not filter[:, 1].all()
        assert filter[:, 2].all()
        assert not filter[:, 3].all()

    @pytest.mark.parametrize("dist_lat", [None, 150.])
    @pytest.mark.parametrize("dist_ax", [None, 300.])
    @pytest.mark.parametrize("dist_vol", [None, 350.])
    def test_filter_kernel_statistical(self, dist_lat, dist_ax, dist_vol):

        """Setup"""
        matcher = match_em.GreedyHungarianMatching(match_dims=2, dist_lat=dist_lat, dist_ax=dist_ax, dist_vol=dist_vol)

        n_out = 1000
        n_tar = 1200
        xyz_out = torch.rand((10, n_out, 3)) * torch.tensor([500, 500, 1000]).unsqueeze(0).unsqueeze(
            0)  # batch implementation
        xyz_tar = torch.rand((10, n_tar, 3)) * torch.tensor([500, 500, 1000]).unsqueeze(0).unsqueeze(
            0)  # batch implementation

        """Run"""
        act = matcher.filter(xyz_out, xyz_tar)  # active pairs

        """Asserts"""
        self.assert_dists(xyz_out, xyz_tar, dist_lat, dist_ax, dist_vol, act)

    test_coordinates = [  # xyz_out, xyz_tar, expected outcome
        # 1 Prediction, 1 References
        (torch.tensor([[0., 0., 0.]]), torch.tensor([[0.1, 0.01, 0.]]), (torch.tensor([0]), torch.tensor([0]))),
        # 4 Predictions, 1 References
        (torch.tensor([[-0.5, -0.5, 0.], [0.6, -0.5, 0.], [-0.4, 0.5, 0.], [0.35, 0.35, 0.]]),
         torch.tensor([[0., 0., 0.]]),
         (torch.tensor([3]), torch.tensor([0]))),
        # 1 Predictions, 4 References
        (torch.tensor([[0., 0., 0.]]),
         torch.tensor([[-0.5, -0.5, 0.], [0.6, -0.5, 0.], [-0.4, 0.5, 0.], [0.35, 0.35, 0.]]),
         (torch.tensor([0]), torch.tensor([3]))),
        # 0 Predictions, 0 References
        (torch.zeros((0, 3)), torch.zeros((0, 3)), (torch.tensor([]), torch.tensor([])))
    ]

    @pytest.mark.parametrize("match_dims", [2, 3])
    @pytest.mark.parametrize("xyz_out,xyz_tar,expected", test_coordinates)
    def test_match_kernel(self, match_dims, xyz_out, xyz_tar, expected):
        """Setup"""
        matcher = match_em.GreedyHungarianMatching(match_dims=match_dims,
                                                   dist_lat=1.,
                                                   dist_ax=2.,
                                                   dist_vol=None)

        """Run"""
        filter_mask = matcher.filter(xyz_out.unsqueeze(0), xyz_tar.unsqueeze(0))
        assignment = matcher._match_kernel(xyz_out, xyz_tar, filter_mask.squeeze(0))

        tp_ix_out, tp_match_ix_out = assignment[2:]
        tp_ix_exp, tp_match_ix_exp = expected

        """Assert"""
        assert (tp_ix_out.nonzero() == tp_ix_exp).all()  # boolean index in output
        assert (tp_match_ix_out.nonzero() == tp_match_ix_exp).all()

    test_data_forward = [
        (torch.arange(5).unsqueeze(1).float() * 2 + torch.zeros((5, 3)),
         torch.arange(4, -1, -1).unsqueeze(1).float() * 2 + torch.zeros((5, 3)) + torch.rand((5, 1)) - 0.5)
    ]

    @pytest.mark.parametrize("xyz_tar,xyz_out", test_data_forward)
    def test_forward(self, matcher, xyz_tar, xyz_out):
        """Tests the sanity"""

        """Setup"""
        matcher.dist_lat = 1
        em_tar = CoordinateOnlyEmitter(xyz_tar, xy_unit='nm')
        em_out = CoordinateOnlyEmitter(xyz_out, xy_unit='nm')

        """Run"""
        tp, fp, fn, tp_match = matcher.forward(em_out, em_tar)

        """Assertions"""
        assert len(tp) == len(tp_match)
        assert len(tp) + len(fp) == len(em_out)
        assert len(tp) + len(fn) == len(em_tar)

        assert ((tp.xyz - tp_match.xyz) <= 1.).all()
        assert (tp.id == tp_match.id).all()

    def test_forward_statistical(self, matcher):

        matcher.dist_lat = 1.
        xyz_tar = torch.zeros((1000, 3))
        xyz_out = torch.zeros_like(xyz_tar)
        xyz_out[:, 0] = torch.randn_like(xyz_out[:, 0])

        em_tar = CoordinateOnlyEmitter(xyz_tar, xy_unit='nm')
        em_out = CoordinateOnlyEmitter(xyz_out, xy_unit='nm')

        """Run"""
        tp, fp, fn, tp_match = matcher.forward(em_out, em_tar)

        """Assert"""
        assert len(tp) / len(em_tar) == pytest.approx(0.7, abs=0.1)
