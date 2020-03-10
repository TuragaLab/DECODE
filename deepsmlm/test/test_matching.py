import warnings
from unittest.case import TestCase

import pytest
import torch

import deepsmlm.evaluation
import deepsmlm.evaluation.match_emittersets as match_em
import deepsmlm.generic
from deepsmlm import EmitterSet
from deepsmlm.evaluation.match_emittersets import NNMatching


class TestMatcherABC:
    """
    Defines tests that should succeed on all implementations of a matching algorithm. All test classes should
    therefore inherit this test.
    """

    @pytest.fixture()
    def matcher(self):
        class MockMatch(deepsmlm.evaluation.match_emittersets.MatcherABC):
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
        em = deepsmlm.generic.emitter.RandomEmitterSet(1000)
        em.frame_ix = torch.randint_like(em.frame_ix, 100)

        return em

    @pytest.fixture()
    def can_em_tar(self, can_em_out):  # candidate emitter target
        em = deepsmlm.generic.emitter.RandomEmitterSet(len(can_em_out))
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

    @pytest.mark.parametrize("dist_lat", [None, 1.])
    @pytest.mark.parametrize("dist_ax", [None, 1.])
    @pytest.mark.parametrize("dist_vol", [None, 1.])
    def test_filter_kernel(self, dist_lat, dist_ax, dist_vol):

        """Setup"""
        matcher = match_em.GreedyHungarianMatching(match_dims=2, dist_lat=dist_lat, dist_ax=dist_ax, dist_vol=dist_vol)

        n_out = 100
        n_tar = 120
        xyz_out = torch.rand((1, n_out, 3))  # batch implementation
        xyz_tar = torch.rand((1, n_tar, 3))  # batch implementation

        """Run"""
        act = matcher.filter(xyz_out, xyz_tar)  # active pairs

        """Asserts"""
        if dist_lat is not None:
            dist_mat = torch.cdist(xyz_out[:, :, :2], xyz_tar[:, :, :2], p=2).sqrt()
            assert (dist_mat[act] <= dist_lat).all()

        if dist_ax is not None:
            dist_mat = torch.cdist(xyz_out[:, :, [2]], xyz_tar[:, :, [2]], p=2).sqrt()
            assert (dist_mat[act] <= dist_ax).all()

        if dist_vol is not None:
            dist_mat = torch.cdist(xyz_out, xyz_tar, p=2).sqrt()
            assert (dist_mat[act] <= dist_vol).all()

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

        tp_ix_out, tp_match_ix_out = assignment
        tp_ix_exp, tp_match_ix_exp = expected

        """Assert"""
        assert (tp_ix_out.nonzero() == tp_ix_exp).all()  # boolean index in output
        assert (tp_match_ix_out.nonzero() == tp_match_ix_exp).all()


#
# class TestNNMatching(TestMatcherABC):
#
#     @pytest.fixture()
#     def matcher(self):
#         return NNMatching()