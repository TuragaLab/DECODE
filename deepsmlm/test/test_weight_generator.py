import torch
import pytest
import numpy as np

from deepsmlm.generic.utils import test_utils
from deepsmlm.neuralfitter import weight_generator
from deepsmlm.generic import emitter


class TestOneHot2ROI:

    @pytest.fixture(params=['zero', 'mean'])
    def cand(self, request):
        return weight_generator.OneHotInflator(roi_size=3, channels=1, overlap_mode=request.param)

    def test_sanity(self, cand):

        with pytest.raises(NotImplementedError):
            cand.__init__(roi_size=3, channels=1, overlap_mode='median')  # wrong overlap mode

        with pytest.raises(NotImplementedError):
            cand.__init__(roi_size=5, channels=1, overlap_mode='zero')  # not implemented roi size

    def test_is_overlap(self, cand):
        """

        Args:
            cand: fixture

        """

        """Setup"""
        x = torch.zeros((2, 1, 32, 32))
        x[0, 0, 2, 2] = 1.  # isolated
        x[1, 0, 10, 10] = 1.  # close by and overlapped
        x[1, 0, 11, 11] = 1.

        """Run"""
        is_overlap, count = cand._is_overlap(x)

        """Assertions"""
        assert is_overlap.dtype is torch.bool
        assert count.dtype in (torch.int16, torch.int32, torch.int64)

        assert x.size() == is_overlap.size()
        assert is_overlap.size() == count.size()

        assert (count[0].unique() == torch.tensor([0, 1])).all()
        assert (count[1].unique() == torch.tensor([0, 1, 2])).all()

        assert not is_overlap[0, 0, 2, 2]
        assert count[0, 0, 2, 2] == 1
        assert (count[1, 0, [10, 11], [10, 11]] == 2).all()

    def test_forward(self, cand):

        """Setup"""
        x = torch.zeros((2, 1, 32, 32))
        x[0, 0, 2, 2] = 1.  # isolated
        x[1, 0, 10, 10] = 1.  # close by and overlapped
        x[1, 0, 11, 11] = 1.

        """Run"""
        xrep = cand.forward(x)

        """Assertions"""
        assert xrep.size() == x.size()

        # Centres
        assert x[0, 0, 2, 2] == xrep[0, 0, 2, 2]
        assert x[1, 0, 10, 10] == xrep[1, 0, 10, 10]
        assert x[1, 0, 11, 11] == xrep[1, 0, 11, 11]

        # non-overlapping parts
        assert (xrep[0, 0, 1:3, 1:3] == 1.).all()
        assert (xrep[1, 0, 9:12, 9] == 1.).all()
        assert (xrep[1, 0, 9, 9:12] == 1.).all()

        # overlapping parts
        if cand.overlap_mode == 'zero':
            assert (xrep[1, 0, [10, 11], [11, 10]] == 0).all()
        elif cand.overlap_mode == 'mean':
            assert (xrep[1, 0, [10, 11], [11, 10]] == 1.).all()
        else:
            raise ValueError


class TestWeightGenerator:

    @pytest.fixture()
    def waiter(self):  # a pun
        class WeightGeneratorMock(weight_generator.WeightGenerator):
            def forward(self, x, y, z):
                x = super().forward(x, y, z)
                return self._forward_return_original(torch.ones_like(x))

        return WeightGeneratorMock()

    def test_shape(self, waiter):
        """

        Args:
            waiter: fixture

        """

        """Setup"""
        x = torch.rand((3, 6, 32, 32))

        """Run"""
        out = waiter.forward(x, None, None)

        """Assertions"""
        # Check shape. Note that the channel dimensions might different.
        assert x.size(0) == out.size(0)
        assert x.size(-1) == out.size(-1)
        assert x.size(-2) == out.size(-2)

    def test_dim_handling(self, waiter):
        """

        Args:
            waiter: fixture

        """
        """Setup"""
        x = torch.rand((6, 32, 32))

        """Run"""
        out = waiter.forward(x, None, None)

        """Assertions"""
        assert out.dim() == x.dim()

        with pytest.raises(ValueError):
            _ = waiter.forward(torch.rand((32, 32)), None, None)

        with pytest.raises(ValueError):
            _ = waiter.forward(torch.rand((2, 3, 2, 32, 32)), None, None)


class TestSimpleWeight(TestWeightGenerator):

    @pytest.fixture(scope='class')
    def waiter(self):
        return weight_generator.SimpleWeight((-0.5, 4.5), (-0.5, 4.5), (5, 5), 3, 6, 'constant')

    def test_const_weight(self, weighter):
        em = emitter.CoordinateOnlyEmitter(torch.tensor([[1., 1., 0], [3., 3., 0.]]))
        mask = weighter.forward(torch.zeros((1, 5, 5)), em, None)
        assert test_utils.tens_almeq(mask[[0, 5]], torch.ones_like(mask[[0, 5]]))  # p channel and bg are all weight 1
        mask_tar = torch.ones((5, 5))
        mask_tar[2, 2] = 0.
        mask_tar[3:, :2] = 0.
        mask_tar[:2, 3:] = 0.
        for i in [1, 2, 3, 4]:
            assert test_utils.tens_almeq(mask[i], mask_tar)



class TestDerivePseudobgFromBg:

    @pytest.fixture(scope='class')
    def pseudobg(self):
        return weight_generator.DerivePseudobgFromBg((-0.5, 63.5), (-0.5, 63.5), (64, 64), [17, 17], 8)

    # @pytest.fixture(scope='class')
    def bg_sampler(self):
        em = emitter.RandomEmitterSet(20, 64)
        fix = np.linspace(0, 3, 20).round()
        np.random.shuffle(fix)
        em.frame_ix = torch.from_numpy(fix)
        bg = torch.rand((5, 1, 64, 64))

        return em, bg

    def test_candidate(self, pseudobg):
        em, bg = self.bg_sampler()
        bg[2] *= 100  # test whether this comes out

        pseudobg.forward(None, em, bg)
        em_c = em.get_subset_frame(2, 2)
        assert test_utils.tens_almeq(em_c.bg, torch.ones_like(em_c.bg) * bg[2].mean(), 5.)


class TestCRLBWeight:

    @pytest.fixture(scope='class')
    def pseudo_em(self):
        em = emitter.RandomEmitterSet(20, 64)
        em.bg = torch.rand_like(em.bg)
        em.xyz_cr = torch.rand_like(em.xyz_cr)
        em.phot_cr = torch.rand_like(em.phot_cr)
        em.bg_cr = torch.rand_like(em.bg_cr)

        return em

    @pytest.fixture(scope='class')
    def generator(self):
        return weight_generator.GenerateWeightMaskFromCRLB((-0.5, 63.5), (-0.5, 63.5), (64, 64), 3)

    def test_candidate(self, generator, pseudo_em):
        generator.forward(torch.rand((3, 3, 64, 64)), pseudo_em, None)
        print("Done.")



