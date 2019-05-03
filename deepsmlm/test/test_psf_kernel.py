import torch
from unittest import TestCase

from deepsmlm.generic.emitter import CoordinateOnlyEmitter
from deepsmlm.generic.psf_kernel import DeltaPSF, OffsetPSF


class TestOffsetPSF(TestCase):
    def setUp(self) -> None:
        """
        Implicit test on the constructor
        Do not change this here, because then the tests will be broken.
        """
        self.psf_bin_1px = OffsetPSF((-0.5, 31.5),
                                     (-0.5, 31.5),
                                     (32, 32))

        self.delta_psf_1px = DeltaPSF((-0.5, 31.5),
                                  (-0.5, 31.5),
                                  None, (32, 32), 0, False, 0)

        self.psf_bin_halfpx = OffsetPSF((-0.5, 31.5),
                                        (-0.5, 31.5),
                                        (64, 64))

        self.delta_psf_hpx = DeltaPSF((-0.5, 31.5),
                                      (-0.5, 31.5),
                                      None, (64, 64), 0, False, 0)

    def test_bin_centrs(self):
        """
        Test the bin centers.
        :return:
        """
        self.assertEqual(-0.5, self.psf_bin_1px.bin_x[0])
        self.assertEqual(0.5, self.psf_bin_1px.bin_x[1])
        self.assertEqual(0., self.psf_bin_1px.bin_ctr_x[0])
        self.assertEqual(0., self.psf_bin_1px.bin_ctr_y[0])

        self.assertEqual(-0.5, self.psf_bin_halfpx.bin_x[0])
        self.assertEqual(0., self.psf_bin_halfpx.bin_x[1])
        self.assertEqual(-0.25, self.psf_bin_halfpx.bin_ctr_x[0])
        self.assertEqual(-0.25, self.psf_bin_halfpx.bin_ctr_y[0])

    def test_offset_range(self):

        self.assertEqual(0.5, self.psf_bin_1px.offset_max_x)
        self.assertEqual(0.5, self.psf_bin_1px.offset_max_x)
        self.assertEqual(0.25, self.psf_bin_halfpx.offset_max_y)
        self.assertEqual(0.25, self.psf_bin_halfpx.offset_max_y)

    def test_foward_range(self):

        xyz = CoordinateOnlyEmitter(torch.rand((1000, 3)) * 40)
        offset_1px = self.psf_bin_1px.forward(xyz)
        offset_hpx = self.psf_bin_halfpx.forward(xyz)

        self.assertTrue(offset_1px.max().item() <= 0.5)
        self.assertTrue(offset_1px.min().item() > -0.5)
        self.assertTrue(offset_hpx.max().item() <= 0.25)
        self.assertTrue(offset_hpx.min().item() > -0.25)

    def test_forward_indexing_hc(self):
        """
        Test whether delta psf and offset map share the same indexing (i.e. the order of the axis
        is consistent).
        :return:
        """
        xyz = CoordinateOnlyEmitter(torch.tensor([[15.1, 2.9, 0.]]))

        img_nonzero = self.delta_psf_1px.forward(xyz)[0].nonzero()
        self.assertTrue(torch.equal(img_nonzero, self.psf_bin_1px.forward(xyz)[0].nonzero()))
        self.assertTrue(torch.equal(img_nonzero, self.psf_bin_1px.forward(xyz)[1].nonzero()))

        img_nonzero = self.delta_psf_hpx.forward(xyz)[0].nonzero()
        self.assertTrue(torch.equal(img_nonzero, self.psf_bin_halfpx.forward(xyz)[0].nonzero()))
        self.assertTrue(torch.equal(img_nonzero, self.psf_bin_halfpx.forward(xyz)[1].nonzero()))

    def test_outofrange(self):
        """
        Test whether delta psf and offset map share the same indexing (i.e. the order of the axis
        is consistent).
        :return:
        """
        xyz = CoordinateOnlyEmitter(torch.tensor([[31.6, 16., 0.]]))
        offset_map = self.psf_bin_1px.forward(xyz)
        self.assertTrue(torch.equal(torch.zeros_like(offset_map), offset_map))

    def test_forward_offset_1px_units(self):
        """
        Test forward with 1 px = 1 unit
        :return:
        """
        xyz = CoordinateOnlyEmitter(torch.tensor([[0., 0., 0.],
                            [1.5, 1.2, 0.],
                            [2.7, 0.5, 0.]]))

        offset_1px = self.psf_bin_1px.forward(xyz)

        self.assertTrue(torch.allclose(torch.tensor([0., 0.]), offset_1px[:, 0, 0]))
        self.assertTrue(torch.allclose(torch.tensor([0.5, 0.2]), offset_1px[:, 1, 1]))
        self.assertTrue(torch.allclose(torch.tensor([-0.3, 0.5]), offset_1px[:, 3, 0]))

    def test_forward_offset_hpx(self):
        xyz = CoordinateOnlyEmitter(torch.tensor([[0., 0., 0.],
                            [0.5, 0.2, 0.]]))

        offset_hpx = self.psf_bin_halfpx.forward(xyz)

        # x_exp = torch.tensor([[0]])
        #
        # self.assertTrue(torch.allclose(torch.tensor([])))
        return True



