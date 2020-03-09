import pytest
import torch
from matplotlib import pyplot as plt

import deepsmlm.simulation.psf_kernel as psf_kernel
from deepsmlm.generic import EmitterSet, CoordinateOnlyEmitter, RandomEmitterSet
from deepsmlm.generic.plotting.frame_coord import PlotFrame, PlotFrameCoord
from deepsmlm.generic.utils import test_utils as tutil
from deepsmlm.generic.utils.test_utils import equal_nonzero
from deepsmlm.neuralfitter.losscollection import OffsetROILoss
from deepsmlm.neuralfitter.target_generator import SinglePxEmbedding, KernelEmbedding, GlobalOffsetRep, SpatialEmbedding


class TestTargetGenerator:

    @pytest.fixture(scope='class')
    def targ(self):
        """
        Setup dummy target generator for inheritors.

        """
        class DummyTarget:
            def __init__(self, xextent, yextent, img_shape):
                self.xextent = xextent
                self.yextent = yextent
                self.img_shape = img_shape

                self.delta = psf_kernel.DeltaPSF(xextent=self.xextent,
                                                 yextent=self.yextent,
                                                 img_shape=self.img_shape)

            def forward(self, x: EmitterSet) -> torch.Tensor:
                return self.delta.forward(x.xyz, x.phot).unsqueeze(1)

        xextent = (-0.5, 63.5)
        yextent = (-0.5, 63.5)
        img_shape = (64, 64)
        return DummyTarget(xextent, yextent, img_shape)

    @pytest.fixture(scope='class')
    def fem(self):
        return EmitterSet(xyz=torch.tensor([[0., 0., 0.]]), phot=torch.Tensor([1.]), frame_ix=torch.tensor([0]))

    def test_shape(self, targ, fem):
        """
        Tests the frame_shape

        Args:
            targ:
            fem:

        """

        out = targ.forward(fem)

        """Tests"""
        assert out.dim() == 4, "Wrong dimensionality."


class TestSpatialEmbedding:

    @pytest.fixture(scope='class')
    def tar_bin_1px(self):
        return SpatialEmbedding((-0.5, 31.5),
                                (-0.5, 31.5),
                                (32, 32))

    @pytest.fixture(scope='class')
    def tar_bin_05px(self):
        return SpatialEmbedding((-0.5, 31.5),
                                (-0.5, 31.5),
                                (64, 64))

    @pytest.fixture(scope='class')
    def delta_1px(self):
        return psf_kernel.DeltaPSF((-0.5, 31.5), (-0.5, 31.5), (32, 32), False, None)

    @pytest.fixture(scope='class')
    def delta_05px(self):
        return psf_kernel.DeltaPSF((-0.5, 31.5), (-0.5, 31.5), (64, 64), False, None)

    def test_binning(self, tar_bin_1px, tar_bin_05px):
        """
        Tests the bins

        Returns:

        """
        assert -0.5 == tar_bin_1px._bin_x[0]
        assert 0.5 == tar_bin_1px._bin_x[1]
        assert 0. == tar_bin_1px._bin_ctr_x[0]
        assert 0. == tar_bin_1px._bin_ctr_y[0]

        assert -0.5 == tar_bin_05px._bin_x[0]
        assert 0. == tar_bin_05px._bin_x[1]
        assert -0.25 == tar_bin_05px._bin_ctr_x[0]
        assert -0.25 == tar_bin_05px._bin_ctr_y[0]

        assert 0.5 == tar_bin_1px._offset_max_x
        assert 0.5 == tar_bin_1px._offset_max_x
        assert 0.25 == tar_bin_05px._offset_max_y
        assert 0.25 == tar_bin_05px._offset_max_y

    def test_forward_range(self, tar_bin_1px, tar_bin_05px):
        em = CoordinateOnlyEmitter(torch.rand((1000, 3)) * 40)
        offset_1px = tar_bin_1px.forward_(em.xyz)
        offset_hpx = tar_bin_05px.forward_(em.xyz)

        assert offset_1px.max().item() <= 0.5
        assert offset_1px.min().item() > -0.5
        assert offset_hpx.max().item() <= 0.25
        assert offset_hpx.min().item() > -0.25

    def test_forward_indexing_hc(self, tar_bin_1px, tar_bin_05px, delta_1px, delta_05px):
        """
        Test whether delta psf and offset map share the same indexing
        """
        xyz = CoordinateOnlyEmitter(torch.tensor([[15.1, 2.9, 0.]]), xy_unit='px').xyz

        img_ = delta_1px.forward(xyz)
        tar_ = tar_bin_1px.forward_(xyz)
        assert torch.equal(img_.nonzero(), tar_[:, 0].nonzero())
        assert torch.equal(img_.nonzero(), tar_[:, 1].nonzero())

        img_ = delta_05px.forward(xyz)
        tar_ = tar_bin_05px.forward_(xyz)
        assert torch.equal(img_.nonzero(), tar_[:, 0].nonzero())
        assert torch.equal(img_.nonzero(), tar_[:, 1].nonzero())

        """Test an out of range emitter."""
        xyz = CoordinateOnlyEmitter(torch.tensor([[31.6, 16., 0.]]), xy_unit='px').xyz
        offset_map = tar_bin_1px.forward_(xyz)
        assert torch.equal(torch.zeros_like(offset_map), offset_map)

    def test_forward_offset_1px_units(self, tar_bin_1px, tar_bin_05px):
        """
        Another value test during implementation

        Args:
            tar_bin_1px: fixture
            tar_bin_05px: fixture

        """
        xyz = CoordinateOnlyEmitter(torch.tensor([[0., 0., 0.],
                                                  [1.5, 1.2, 0.],
                                                  [2.7, 0.5, 0.]])).xyz

        offset_1px = tar_bin_1px.forward_(xyz).squeeze(0)

        assert torch.allclose(torch.tensor([0., 0.]), offset_1px[:, 0, 0])
        assert torch.allclose(torch.tensor([0.5, 0.2]), offset_1px[:, 1, 1])
        assert torch.allclose(torch.tensor([-0.3, 0.5]), offset_1px[:, 3, 0])


class TestSinglePxEmbedding:

    @pytest.fixture(scope='class')
    def offset_rep(self):
        return SinglePxEmbedding(xextent=(-0.5, 31.5), yextent=(-0.5, 31.5), img_shape=(32, 32))

    @pytest.fixture(scope='class')
    def em_set(self):
        num_emitter = 100
        return EmitterSet(torch.rand((num_emitter, 3)) * 40,
                          torch.rand(num_emitter),
                          torch.zeros(num_emitter).int(), xy_unit='px')

    def test_forward_equal_nonzeroness(self, offset_rep, em_set):
        """
        Test whether the non-zero entries are the same in all channels. Assumes that nothing is exactly at 0.0...
        """
        out = offset_rep.forward_(em_set.xyz, em_set.phot, em_set.frame_ix)
        assert equal_nonzero(out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4])

    def test_output_range(self, offset_rep, em_set):
        """
        Test whether delta x/y are between -0.5 and 0.5 (which they need to be for 1 coordinate unit == 1px)
        """
        out = offset_rep.forward_(em_set.xyz, em_set.phot, em_set.frame_ix)
        p, I, dx, dy, z = out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4]

        assert (dx <= 0.5).all(), "delta x/y must be between -0.5 and 0.5"
        assert (dx >= -0.5).all(), "delta x/y must be between -0.5 and 0.5"
        assert (dy <= 0.5).all(), "delta x/y must be between -0.5 and 0.5"
        assert (dy >= -0.5).all(), "delta x/y must be between -0.5 and 0.5"

    def test_single_emitter(self, offset_rep):
        """
        Tests a single emitter for which I know the exact values.
        Args:
            offset_rep: fixture

        """
        em_set = CoordinateOnlyEmitter(torch.tensor([[15.1, 19.6, 250.]]), xy_unit='px')
        out = offset_rep.forward_(em_set.xyz, em_set.phot, em_set.frame_ix)[0]  # single frame
        assert tutil.tens_almeq(out[:, 15, 20], torch.tensor([1., 1., 0.1, -0.4, 250.]), 1e-5)


class TestKernelEmbedding(TestSinglePxEmbedding):

    @pytest.fixture(scope='class')
    def roi_offset(self):
        return KernelEmbedding((-0.5, 31.5), (-0.5, 31.5), (32, 32))

    @pytest.fixture(scope='class')
    def em_set_fixed_values(self):
        """
        Fixture to test hand-calculated values.
        One easy emitter, two adjacent emitters, two overlaying emitters.
        """
        return CoordinateOnlyEmitter(torch.tensor([[14.9, 17.2, 300.],
                                                   [0.01, 0.01, 250.],
                                                   [0.99, 0.99, -250.],
                                                   [25., 25., 500.],
                                                   [25.2, 25.2, 700.],
                                                   [10., 10., 200.],
                                                   [11., 11., 500.]]), xy_unit='px')

    @pytest.fixture(scope='class')
    def loss(self):
        return OffsetROILoss(roi_size=3)

    def test_forward_equal_nonzeroness(self, roi_offset, em_set):
        """
        Test whether the non-zero entries are the same in all channels. Assumes that nothing is exactly at 0.0...
        """
        out = roi_offset.forward_(em_set.xyz, em_set.phot, em_set.frame_ix)
        assert equal_nonzero(out[:, 1], out[:, 2], out[:, 3], out[:, 4])

    def test_values(self, roi_offset, em_set_fixed_values, loss):
        target = roi_offset.forward_(em_set_fixed_values.xyz, em_set_fixed_values.phot, None)[0]  # single frame

        assert tutil.tens_almeq(target[:, 15, 17], torch.tensor([1., 1., -0.1, 0.2, 300.]), 1e-5)
        assert tutil.tens_almeq(target[2, 15, 16:19], torch.tensor([-0.1, -0.1, -0.1]), 1e-5)
        assert tutil.tens_almeq(target[3, 14:17, 17], torch.tensor([0.2, 0.2, 0.2]), 1e-5)


@pytest.mark.skip("Deprecated.")
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

        offset_maps = classyclassclass.forward_(em)
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
        offset_maps = classyclassclass.forward_(em)
        PlotFrameCoord(frame=offset_maps[1], pos_tar=em.xyz).plot()
        plt.show()

        PlotFrameCoord(frame=offset_maps[2], pos_tar=em.xyz).plot()
        plt.show()

        PlotFrameCoord(frame=offset_maps[3], pos_tar=em.xyz).plot()
        plt.show()

        PlotFrameCoord(frame=offset_maps[4], pos_tar=em.xyz).plot()
        plt.show()
        assert True
