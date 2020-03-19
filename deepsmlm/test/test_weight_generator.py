import pytest
import torch

from deepsmlm.generic import emitter
from deepsmlm.generic.utils import test_utils
from deepsmlm.neuralfitter import weight_generator


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
        x = torch.rand((3, 6, 5, 5))
        em = emitter.EmptyEmitterSet()
        opt = torch.rand_like(x[:, [0]])

        """Run"""
        out = waiter.forward(x, em, opt)

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
        x = torch.rand((6, 5, 5))
        em = emitter.EmptyEmitterSet()

        """Run"""
        out = waiter.forward(x, em, None)

        """Assertions"""
        assert out.dim() == x.dim()

        with pytest.raises(ValueError):
            _ = waiter.forward(torch.rand((5, 5)), None, None)

        with pytest.raises(ValueError):
            _ = waiter.forward(torch.rand((2, 3, 2, 5, 5)), None, None)


class TestSimpleWeight(TestWeightGenerator):

    @pytest.fixture(scope='class', params=[('const', None), ('phot', 2.3)])
    def waiter(self, request):
        return weight_generator.SimpleWeight(xextent=(-0.5, 4.5),
                                             yextent=(-0.5, 4.5),
                                             img_shape=(5, 5),
                                             target_roi_size=3,
                                             weight_mode=request.param[0],
                                             weight_power=request.param[1])

    def test_sanity(self, waiter):

        """Test init sanity"""
        with pytest.raises(ValueError):  # const and weight power != 1
            weight_generator.SimpleWeight(xextent=(-0.5, 4.5), yextent=(-0.5, 4.5), img_shape=(5, 5),
                                          target_roi_size=3, weight_mode='const', weight_power=2.3)

        with pytest.raises(ValueError):
            weight_generator.SimpleWeight(xextent=(-0.5, 4.5), yextent=(-0.5, 4.5), img_shape=(5, 5),
                                          target_roi_size=3, weight_mode='a', weight_power=None)

        """Test forward sanity"""
        with pytest.raises(ValueError):  # wrong spatial dim
            waiter.forward(torch.zeros((1, 6, 6, 6)), emitter.EmptyEmitterSet(), None)

        with pytest.raises(ValueError):  # wrong channel count
            waiter.forward(torch.zeros((1, 3, 5, 5)), emitter.EmptyEmitterSet(), None)

        with pytest.raises(ValueError):
            em = emitter.RandomEmitterSet(32)
            em.phot[5] = 0.
            waiter.forward(torch.rand((1, 6, 5, 5)), em, None)

        if waiter.weight_mode == 'phot':
            with pytest.raises(ValueError):  # bg has zero values
                waiter.forward(torch.zeros((1, 6, 5, 5)), emitter.EmptyEmitterSet(), None)

    def test_weight_hard(self, waiter):
        """
        Tests entire weight unit with hard computed values

        Args:
            waiter: fixture

        """

        """Setup"""
        tar_frames = torch.zeros((1, 6, 5, 5))
        tar_frames[:, 5] = torch.rand_like(tar_frames[:, 5])  # let bg be non-zero

        em = emitter.EmitterSet(xyz=torch.tensor([[1., 1., 0], [3., 3., 0.]]), phot=torch.Tensor([1., 5.]),
                                frame_ix=torch.tensor([0, 1]))

        """Run"""
        mask = waiter.forward(tar_frames, em, None)

        """Assertions"""
        assert (mask[:, 0] == 1.).all(), "p channel must be weight 1"
        # some zero value assertions applicaple for both const and phot weight
        assert (mask[:, 1:-1, 2, 2] == 0.).all(), "intersection must be 0"
        assert (mask[:, 1:-1, 3:, :2] == 0.).all()
        assert (mask[:, 1:-1, :2, 3:] == 0.).all()

        if waiter.weight_mode == 'const':
            assert (mask[:, 5] == 1.).all(), "bg channel must be weight 1"
            assert (mask[:, 1:-1].unique() == torch.tensor([0., 1])).all(), "in const. mode values must be 0 or 1"

        if waiter.weight_mode == 'phot':
            assert (mask[:, 1:-1, :2, :2] == 1.).all(), "CRLB estimate for photon count 1"
            assert mask[:, 1, 3, 3] == pytest.approx(0.02468, abs=0.0001), "Photon CRLB estimate for count of 5"
            assert mask[:, 2, 3, 3] == pytest.approx(40.51641, abs=0.0001), "X CRLB estimate"
            assert mask[:, 3, 3, 3] == pytest.approx(40.51641, abs=0.0001), "Y CRLB estimate"
            assert mask[:, 4, 3, 3] == pytest.approx(40.51641, abs=0.0001), "Y CRLB estimate"
            assert test_utils.tens_almeq(mask[:, 5], 1 / tar_frames[:, 5] ** 2.3, 1e-5), "BG CRLB estimate"
