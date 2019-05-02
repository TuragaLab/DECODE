from unittest import TestCase
import torch

from deepsmlm.generic.emitter import EmitterSet
from deepsmlm.generic.psf_kernel import DeltaPSF
from deepsmlm.neuralfitter.pre_processing import ZasOneHot, DecodeRepresentation


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


class TestDecodeRepresentation(TestCase):
    def setUp(self) -> None:
        num_emitter = 100
        self.em = EmitterSet(torch.rand((num_emitter, 3)) * 40,
                             torch.rand(num_emitter),
                             torch.zeros(num_emitter))

        self.dc = DecodeRepresentation((-0.5, 31.5), (-0.5, 31.5), None, (32, 32))

    def test_forward_shape(self):

        _, p, I, x, y, z = self.dc.forward(self.em)
        self.assertEqual(p.shape, I.shape, "Maps of Decode Rep. must have equal dimension.")
        self.assertEqual(I.shape, x.shape, "Maps of Decode Rep. must have equal dimension.")
        self.assertEqual(x.shape, y.shape, "Maps of Decode Rep. must have equal dimension.")
        self.assertEqual(y.shape, z.shape, "Maps of Decode Rep. must have equal dimension.")

    def test_forward_equal_nonzeroness(self):
        """
        Test whether the non-zero entries are the same in all channels.
        Note that this in principle stochastic but the chance of having random float exactly == 0 is super small.
        :return:
        """
        _, p, I, x, y, z = self.dc.forward(self.em)
        self.assertTrue(equal_nonzero(p, I, x, y, z))

