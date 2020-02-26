import pytest
import torch
from matplotlib import pyplot as plt

from deepsmlm.generic import EmitterSet, CoordinateOnlyEmitter, RandomEmitterSet
from deepsmlm.generic.plotting.frame_coord import PlotFrame, PlotFrameCoord
from deepsmlm.generic.utils import test_utils as tutil
from deepsmlm.neuralfitter.losscollection import OffsetROILoss
from deepsmlm.neuralfitter.target_generator import OffsetRep, ROIOffsetRep, GlobalOffsetRep, OffsetPSF
from deepsmlm.generic.utils.test_utils import equal_nonzero
import deepsmlm.generic.psf_kernel as psf_kernel


class TestOffsetPSF(TestCase):
    def setUp(self) -> None:
        """
        Implicit test on the constructor
        Do not change this here, because then the tests will be broken.
        """
        self.psf_bin_1px = OffsetPSF((-0.5, 31.5),
                                     (-0.5, 31.5),
                                     (32, 32))

        self.delta_psf_1px = psf_kernel.DeltaPSF((-0.5, 31.5),
                                                 (-0.5, 31.5),
                                                 None, (32, 32), 0, False, 0)

        self.psf_bin_halfpx = OffsetPSF((-0.5, 31.5),
                                        (-0.5, 31.5),
                                        (64, 64))

        self.delta_psf_hpx = psf_kernel.DeltaPSF((-0.5, 31.5),
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


class TestDecodeRepresentation:

    @pytest.fixture(scope='class')
    def offset_rep(self):
        return OffsetRep(xextent=(-0.5, 31.5), yextent=(-0.5, 31.5), zextent=None, img_shape=(32, 32), cat_output=False)

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

    def test_single_emitter(self, offset_rep):
        em = CoordinateOnlyEmitter(torch.tensor([[15.1, 19.6, 250.]]))
        offset_rep.cat_out = True
        target = offset_rep.forward(em)
        assert tutil.tens_almeq(target[:, 15, 20], torch.tensor([1., 1., 0.1, -0.4, 250.]), 1e-5)


class TestROIOffsetRep(TestDecodeRepresentation):

    @pytest.fixture(scope='class')
    def roi_offset(self):
        return ROIOffsetRep((-0.5, 31.5), (-0.5, 31.5), None, (32, 32))

    @pytest.fixture(scope='class')
    def em_set(self):
        """An easy emitter, two adjacent emitters, two overlaying emitters"""
        return CoordinateOnlyEmitter(torch.tensor([[14.9, 17.2, 300.],
                                                   [0.01, 0.01, 250.],
                                                   [0.99, 0.99, -250.],
                                                   [25., 25., 500.],
                                                   [25.2, 25.2, 700.],
                                                   [10., 10., 200.],
                                                   [11., 11., 500.]]))

    @pytest.fixture(scope='class')
    def loss(self):
        return OffsetROILoss(roi_size=3)

    def test_values(self, roi_offset, em_set, loss):
        target = roi_offset.forward(em_set)

        # plt.figure(figsize=(12, 8))
        # plt.subplot(231)
        # PlotFrame(target[0]).plot()
        # plt.subplot(232)
        # PlotFrame(target[1]).plot()
        # plt.subplot(234)
        # PlotFrame(target[2]).plot()
        # plt.subplot(235)
        # PlotFrame(target[3]).plot()
        # plt.subplot(236)
        # PlotFrame(target[4]).plot()
        # plt.show()

        assert tutil.tens_almeq(target[:, 15, 17], torch.tensor([1., 1., -0.1, 0.2, 300.]), 1e-5)
        assert tutil.tens_almeq(target[2, 15, 16:19], torch.tensor([-0.1, -0.1, -0.1]), 1e-5)
        assert tutil.tens_almeq(target[3, 14:17, 17], torch.tensor([0.2, 0.2, 0.2]), 1e-5)

        """Test it together with the loss"""
        prediction = torch.rand((1, 5, 32, 32))
        loss_v = loss(prediction, target.unsqueeze(0))
        assert tutil.tens_almeq(loss_v[0, 1:, 1, 0], torch.zeros(4))
        assert tutil.tens_almeq(loss_v[0, 1:, 0, 1], torch.zeros(4))


class TestGlobalOffsetRep:

    @pytest.fixture(scope='class')
    def classyclassclass(self):
        return GlobalOffsetRep((-0.5, 31.5), (-0.5, 31.5), None, (32, 32))

    @pytest.fixture(scope='class')
    def two_em(self):
        return CoordinateOnlyEmitter(torch.tensor([[-0.5, 15.5, 0.], [31.5, 15.5, 0.]]))

    def test_forward_shape(self, classyclassclass, two_em):
        assert classyclassclass.assign_emitter(two_em).shape == torch.zeros(32, 32).shape

    def test_classification(self, classyclassclass, two_em):
        gt = torch.zeros(32, 32).type(torch.LongTensor)
        gt[16:, :] = 1

        prediction_ix = classyclassclass.assign_emitter(two_em)
        assert tutil.tens_almeq(gt, prediction_ix)

    @pytest.mark.skip("Only for plotting.")
    def test_diag_em(self, classyclassclass):
        em = CoordinateOnlyEmitter(torch.tensor([[0., 0., 0.], [31., 31., 0.]]))
        prediction_ix = classyclassclass.assign_emitter(em)
        prediction_ix = prediction_ix.reshape(classyclassclass.img_shape[0], classyclassclass.img_shape[1])

        PlotFrame(prediction_ix).plot()
        plt.show()

        em = RandomEmitterSet(10, extent=32)
        ix_map = classyclassclass.assign_emitter(em)
        ix_map = ix_map.reshape(classyclassclass.img_shape[0], classyclassclass.img_shape[1])
        PlotFrameCoord(frame=ix_map, pos_tar=em.xyz).plot()
        plt.show()

        offset_maps = classyclassclass.forward(em)
        PlotFrameCoord(frame=offset_maps[2], pos_tar=em.xyz).plot()
        plt.show()

        PlotFrameCoord(frame=offset_maps[3], pos_tar=em.xyz).plot()
        plt.show()

        PlotFrameCoord(frame=offset_maps[4], pos_tar=em.xyz).plot()
        plt.show()
        assert True

    @pytest.mark.skip("Only for plotting.")
    def test_mask(self, classyclassclass):
        classyclassclass.masked = True

        em = RandomEmitterSet(10, extent=32)
        em.phot = torch.randn_like(em.phot)
        offset_maps = classyclassclass.forward(em)
        PlotFrameCoord(frame=offset_maps[1], pos_tar=em.xyz).plot()
        plt.show()

        PlotFrameCoord(frame=offset_maps[2], pos_tar=em.xyz).plot()
        plt.show()

        PlotFrameCoord(frame=offset_maps[3], pos_tar=em.xyz).plot()
        plt.show()

        PlotFrameCoord(frame=offset_maps[4], pos_tar=em.xyz).plot()
        plt.show()
        assert True
