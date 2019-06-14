from unittest import TestCase
import torch
import pytest

from deepsmlm.generic.emitter import EmitterSet
from deepsmlm.generic.psf_kernel import DeltaPSF
from deepsmlm.neuralfitter.pre_processing import ZasOneHot, OffsetRep


def equal_nonzero(*a):
    """
    Test whether a and b have the same non-zero elements
    :param a: tensors
    :return: "torch.bool"
    """
    is_equal = torch.equal(a[0], a[0])
    for i in range(a.__len__() - 1):
        is_equal = is_equal * torch.equal(a[i].nonzero(), a[i + 1].nonzero())

    return is_equal


class TestZasMask(TestCase):
    def setUp(self) -> None:
        self.delta_psf = DeltaPSF(xextent=(-0.5, 31.5),
                                  yextent=(-0.5, 31.5),
                                  zextent=None,
                                  img_shape=(32, 32),
                                  dark_value=0.)

        self.zasmask = ZasOneHot(self.delta_psf, 5, 0.8)

    def test_forward(self):
        em = EmitterSet(torch.tensor([[15., 15., 100.]]),
                        torch.tensor([1.]),
                        frame_ix=torch.tensor([0]))
        img = self.zasmask.forward(em)
        self.assertAlmostEqual(1., img.max().item(), places=3, msg="Scaling is wrong.")


class TestDecodeRepresentation:

    @pytest.fixture(scope='class')
    def offset_rep(self):
        return OffsetRep((-0.5, 31.5), (-0.5, 31.5), None, (32, 32), cat_output=False)

    @pytest.fixture(scope='class')
    def em_set(self):
        num_emitter = 100
        return EmitterSet(torch.rand((num_emitter, 3)) * 40,
                          torch.rand(num_emitter),
                          torch.zeros(num_emitter))

    def test_forward_shape(self, offset_rep, em_set):

        p, I, x, y, z = offset_rep.forward(em_set)
        assert p.shape == I.shape, "Maps of Decode Rep. must have equal dimension."
        assert I.shape == x.shape, "Maps of Decode Rep. must have equal dimension."
        assert x.shape == y.shape, "Maps of Decode Rep. must have equal dimension."
        assert y.shape == z.shape, "Maps of Decode Rep. must have equal dimension."

    def test_forward_equal_nonzeroness(self, offset_rep, em_set):
        """
        Test whether the non-zero entries are the same in all channels.
        Note that this in principle stochastic but the chance of having random float exactly == 0 is super small.
        :return:
        """
        p, I, x, y, z = offset_rep.forward(em_set)
        assert equal_nonzero(p, I, x, y, z)

    def test_output_range(self, offset_rep, em_set):
        """
        Test whether delta x/y are between -0.5 and 0.5 (which they need to be for 1 coordinate unit == 1px
        :param offset_rep:
        :param em_set:
        :return:
        """
        p, I, dx, dy, z = offset_rep.forward(em_set)
        assert (dx <= 0.5).all(), "delta x/y must be between -0.5 and 0.5"
        assert (dx >= -0.5).all(), "delta x/y must be between -0.5 and 0.5"
        assert (dy <= 0.5).all(), "delta x/y must be between -0.5 and 0.5"
        assert (dy >= -0.5).all(), "delta x/y must be between -0.5 and 0.5"



