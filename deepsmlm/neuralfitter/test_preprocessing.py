from unittest import TestCase
import torch

from deepsmlm.generic.emitter import EmitterSet
from deepsmlm.generic.psf_kernel import DeltaPSF
from deepsmlm.neuralfitter.pre_processing import ZasOneHot


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
