import pytest
import torch
from abc import ABC

from decode.generic import emitter, test_utils
from decode.neuralfitter import weight_generator


class AbstractWeightGeneratorVerification(ABC):

    def test_check_forward_sanity(self, waiter):

        with pytest.raises(ValueError) as err_info:
            waiter.check_forward_sanity(emitter.EmptyEmitterSet, torch.rand((2, 2)), 0, 0)
            assert err_info == "Unsupported shape of input."

    def test_shape(self, waiter):
        """

        Args:
            waiter: fixture

        """

        """Setup"""
        x = torch.rand((3, 6, 5, 5))
        em = emitter.EmptyEmitterSet(xy_unit='px')
        opt = torch.rand_like(x[:, [0]])

        """Run"""
        out = waiter.forward(em, x, 0, 2)

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
        x = torch.rand((1, 6, 5, 5))
        em = emitter.EmptyEmitterSet(xy_unit='px')

        """Run"""
        out = waiter.forward(em, x, 0, 0)

        """Assertions"""
        assert out.dim() == x.dim()

        with pytest.raises(ValueError):
            _ = waiter.forward(em, torch.rand((5, 5)), 0, 0)

        with pytest.raises(ValueError):
            _ = waiter.forward(em, torch.rand((2, 3, 2, 5, 5)), 0, 0)


class TestSimpleWeight(AbstractWeightGeneratorVerification):

    @pytest.fixture(scope='class', params=[('const', None)])  #, ('phot', 2.3)])
    def waiter(self, request):
        return weight_generator.SimpleWeight(xextent=(-0.5, 4.5),
                                             yextent=(-0.5, 4.5),
                                             img_shape=(5, 5),
                                             roi_size=3,
                                             weight_mode=request.param[0],
                                             weight_power=request.param[1])

    def test_sanity(self, waiter):

        """Test init sanity"""
        with pytest.raises(ValueError):  # const and weight power != 1
            weight_generator.SimpleWeight(xextent=(-0.5, 4.5), yextent=(-0.5, 4.5), img_shape=(5, 5),
                                          roi_size=3, weight_mode='const', weight_power=2.3)

        with pytest.raises(ValueError):
            weight_generator.SimpleWeight(xextent=(-0.5, 4.5), yextent=(-0.5, 4.5), img_shape=(5, 5),
                                          roi_size=3, weight_mode='a', weight_power=None)

        """Test forward sanity"""
        with pytest.raises(ValueError):  # wrong spatial dim
            waiter.forward(emitter.EmptyEmitterSet(), torch.zeros((1, 6, 6, 6)), 0, 0)

        with pytest.raises(ValueError):  # wrong channel count
            waiter.forward(emitter.EmptyEmitterSet(), torch.zeros((1, 3, 5, 5)), 0, 0)

        # with pytest.raises(ValueError):  # negative photon count
        #     em = emitter.RandomEmitterSet(32)
        #     em.phot[5] = -0.1
        #     waiter.forward(em, torch.rand((1, 6, 5, 5)), 0, 0)
        #
        # if waiter.weight_mode == 'phot':
        #     with pytest.raises(ValueError):  # bg has zero values
        #         waiter.forward(emitter.EmptyEmitterSet(), torch.zeros((1, 6, 5, 5)))

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
                                frame_ix=torch.tensor([0, 0]), xy_unit='px')

        """Run"""
        mask = waiter.forward(em, tar_frames, 0, 0)

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


@pytest.mark.skip("Not ready implementation.")
class TestFourFoldWeight(AbstractWeightGeneratorVerification):

    @pytest.fixture()
    def waiter(self):
        return weight_generator.FourFoldSimpleWeight(
            xextent=(-0.5, 63.5), yextent=(-0.5, 63.5), img_shape=(64, 64), roi_size=3,
            rim=0.125)

    def test_forward(self, waiter):

        """Setup"""
        tar_frames = torch.rand((2, 21, 64, 64))
        tar_em = emitter.CoordinateOnlyEmitter(torch.tensor([[0., 0., 0.], [0.49, 0., 0.]]), 'px')

        """Run"""
        weight_out = waiter.forward(tar_em, tar_frames, 0, 1)

        """Assertions"""
        assert weight_out.size(1) == 21
