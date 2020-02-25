import torch
import pytest
import numpy as np
import random
import matplotlib.pyplot as plt
from unittest import TestCase

import deepsmlm.neuralfitter.target_generator
from deepsmlm.generic.emitter import CoordinateOnlyEmitter
import deepsmlm.generic.emitter as emc
import deepsmlm.generic.psf_kernel as psf_kernel
import deepsmlm.generic.inout.load_calibration as load_cal
import deepsmlm.generic.utils.test_utils as tutil
import deepsmlm.generic.plotting.frame_coord as plf
import deepsmlm.generic.plotting.plot_utils as plu


class TestPSF:

    @pytest.fixture(scope='class')
    def abs_psf(self):
        """
        Abstract PSF class fixture

        Returns:
            psf_kernel.PSF: abstract PSF
        """
        class PseudoAbsPSF(psf_kernel.PSF):
            def _forward_single_frame(self, xyz: torch.Tensor, weight: torch.Tensor):
                if xyz.numel() >= 1:
                    return torch.ones((32, 32)) * xyz[0, 0]
                else:
                    return torch.zeros((32, 32))

            def forward(self, xyz: torch.Tensor, weight: torch.Tensor, frame_ix: torch.Tensor, ix_low, ix_high):
                super().forward(xyz, weight, frame_ix, ix_low, ix_high)
                return self._forward_single_frame_wrapper(xyz, weight, frame_ix)

        return PseudoAbsPSF(None, None, None, None)

    def test_single_frame_wrapper(self, abs_psf):
        """
        Tests whether the general impl
        Args:
            abs_psf:

        Returns:

        """

        xyz = torch.Tensor([[15., 15., 0.], [0.5, 0.3, 0.]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.Tensor([1, -2]).int()

        frames = abs_psf._forward_single_frame_wrapper(xyz, phot, frame_ix)

        assert frames.dim() == 3
        assert (frames[0] == 0.5).all()
        assert (frames[1:3] == 0).all()
        assert (frames[-1] == 15).all()


class TestGaussianExpect:
    @pytest.fixture(scope='class')
    def normgauss2d(self):
        return psf_kernel.GaussianExpect((-0.5, 63.5), (-0.5, 63.5), None, img_shape=(64, 64), sigma_0=1.5)

    @pytest.fixture(scope='class')
    def normgauss3d(self):
        return psf_kernel.GaussianExpect((-0.5, 63.5), (-0.5, 63.5), (-5000., 5000.), img_shape=(64, 64), sigma_0=1.5)

    def test_norm(self, normgauss2d, normgauss3d):
        xyz = torch.tensor([[32., 32., 0.]])
        phot = torch.tensor([1.])
        assert pytest.approx(normgauss2d.forward(xyz, phot).sum().item(), 0.05) == 1
        assert pytest.approx(normgauss3d.forward(xyz, phot).sum().item(), 0.05) == 1

    def test_peak_weight(self, normgauss2d, normgauss3d):
        normgauss2d.peak_weight = True
        normgauss3d.peak_weight = True

        xyz = torch.tensor([[32., 32., 0.]])
        phot = torch.tensor([1.])
        assert pytest.approx(normgauss2d.forward(xyz, phot).max().item(), 0.05) == 1
        assert pytest.approx(normgauss2d.forward(xyz, phot).max().item(), 0.05) == 1


class TestOffsetPSF(TestCase):
    def setUp(self) -> None:
        """
        Implicit test on the constructor
        Do not change this here, because then the tests will be broken.
        """
        self.psf_bin_1px = deepsmlm.neuralfitter.target_generator.OffsetPSF((-0.5, 31.5),
                                                                            (-0.5, 31.5),
                                                                            (32, 32))

        self.delta_psf_1px = psf_kernel.DeltaPSF((-0.5, 31.5),
                                                 (-0.5, 31.5),
                                                 None, (32, 32), 0, False, 0)

        self.psf_bin_halfpx = deepsmlm.neuralfitter.target_generator.OffsetPSF((-0.5, 31.5),
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


class TestCubicSplinePSF:
    bead_cal = 'assets/bead_cal_for_testing.mat'

    @pytest.fixture(scope='class')
    def psf_cpu(self):
        xextent = (-0.5, 63.5)
        yextent = (-0.5, 63.5)
        img_shape = (64, 64)

        smap_psf = load_cal.SMAPSplineCoefficient(file=self.bead_cal)
        psf = psf_kernel.CubicSplinePSF(xextent=xextent,
                                        yextent=yextent,
                                        img_shape=img_shape,
                                        roi_size=(32, 32),
                                        coeff=smap_psf.coeff,
                                        vx_size=(1., 1., 10),
                                        ref0=smap_psf.ref0,
                                        cuda=False)

        return psf

    @pytest.fixture(scope='class')
    def psf_cuda(self, psf_cpu):
        return psf_cpu.cuda()

    def test_ship(self, psf_cpu, psf_cuda):
        """
        Tests ships to CPU / CUDA
        Args:
            psf_cpu:
            psf_cuda:

        Returns:

        """
        import spline_psf_cuda

        """Test implementation state before"""
        assert isinstance(psf_cpu._spline_impl, spline_psf_cuda.PSFWrapperCPU)
        assert isinstance(psf_cuda._spline_impl, spline_psf_cuda.PSFWrapperCUDA)

        """Test implementation state after"""
        assert isinstance(psf_cpu.cuda()._spline_impl, spline_psf_cuda.PSFWrapperCUDA)
        assert isinstance(psf_cuda.cpu()._spline_impl, spline_psf_cuda.PSFWrapperCPU)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="can not test CUDA against CPU if CUDA is not available.")
    def test_roi_cuda_cpu(self, psf_cpu, psf_cuda):
        """
        Tests approximate equality of CUDA vs CPU implementation for a few ROIs
        Args:
            psf_cpu: psf implementation on CPU
            psf_cuda: psf implementation on CUDA

        Returns:

        """
        n = 1000
        xyz = torch.rand((n, 3))
        xyz[:, :2] += psf_cpu.ref0[:2]
        xyz[:, 2] = xyz[:, 2] * 1000 - 500
        phot = torch.ones((n, ))

        roi_cpu = psf_cpu.forward_rois(xyz, phot)
        roi_cuda = psf_cuda.forward_rois(xyz, phot)

        assert tutil.tens_almeq(roi_cpu, roi_cuda, 1e-7)
        # return

        """Additional Plotting if manual testing (comment out return statement)"""
        rix = random.randint(0, n - 1)
        plt.figure()
        plt.subplot(121)
        plf.PlotFrame(roi_cpu[rix]).plot()
        plt.colorbar()
        plt.title('CPU implementation')
        plt.subplot(122)
        plf.PlotFrame(roi_cuda[rix]).plot()
        plt.colorbar()
        plt.title('CUDA implementation')
        plt.show()

    def test_frame_cuda_cpu(self, psf_cpu, psf_cuda):
        """
        Tests approximate equality of CUDA vs CPU implementation for a few frames

        Args:
            psf_cpu: psf implementation on CPU
            psf_cuda: psf implementation on CUDA

        Returns:

        """
        n = 10000
        xyz = torch.rand((n, 3)) * 64
        xyz[:, 2] = torch.randn_like(xyz[:, 2]) * 1000 - 500
        phot = torch.ones((n,))
        frame_ix = torch.randint(0, 500, size=(n, ))

        frames_cpu = psf_cpu.forward(xyz, phot, frame_ix)
        frames_cuda = psf_cuda.forward(xyz, phot, frame_ix)

        assert tutil.tens_almeq(frames_cpu, frames_cuda, 1e-7)
        return

        """Additional Plotting if manual testing (comment out return statement)."""
        rix = random.randint(0, frame_ix.max().item() - 1)
        plt.figure()
        plt.subplot(121)
        plf.PlotFrame(frames_cpu[rix]).plot()
        plt.colorbar()
        plt.title('CPU implementation')
        plt.subplot(122)
        plf.PlotFrame(frames_cuda[rix]).plot()
        plt.colorbar()
        plt.title('CUDA implementation')
        plt.show()

    def test_derivative_calculation(self, psf_cuda):
        """
        Tests the derivate calculation

        Args:
            psf_cuda:

        Returns:

        """
        n = 1000
        xyz = torch.rand((n, 3))
        xyz[:, :2] += psf_cuda.ref0[:2]
        xyz[:, 2] = xyz[:, 2] * 1000 - 500
        phot = torch.ones((n,)) * 10000
        bg = 50 * torch.ones((n,))

        drv, rois = psf_cuda.derivative(xyz, phot, bg)

        assert True

    def test_fisher(self, psf_cuda):
        xyz = torch.Tensor([[13., 13., 0.]])
        phot = torch.ones_like(xyz[:, 0]) * 1000
        bg = torch.ones_like(phot) * 50

        fisher, rois = psf_cuda.fisher(xyz, phot, bg)

        assert True

    def test_crlb(self, psf_cuda):
        xyz = torch.Tensor([[13., 13., 0.]])
        phot = torch.ones_like(xyz[:, 0]) * 10000
        bg = torch.ones_like(phot) * 50

        crlb, rois = psf_cuda.crlb(xyz, phot, bg, torch.pinverse)

        assert True


class TestDeprSplinePSF:
    bead_cal = 'assets/bead_cal_for_testing.mat'

    @pytest.fixture(scope='class')
    def psf(self):
        xextent = (-0.5, 63.5)
        yextent = (-0.5, 63.5)
        img_shape = (64, 64)
        psf = load_cal.SMAPSplineCoefficient(self.bead_cal, psf_kernel.DeprCubicSplinePSF).init_spline(xextent, yextent,
                                                                                                       img_shape)

        return psf

    def test_crlb_one_em(self, psf):
        em = emc.CoordinateOnlyEmitter(torch.tensor([[32., 32., 0.]]))
        em.phot = torch.tensor([5000.])
        em.bg = torch.tensor([10.])

        em.populate_crlb(psf)
        assert tutil.tens_seq(em.xyz_scr, torch.tensor([[0.1, 0.1, 1.5]]))
        assert tutil.tens_seq(em.phot_cr, torch.tensor([[200.]]))
        assert not torch.isnan(em.bg_cr).any().item()

    def test_crlb_multi(self, psf):
        em = emc.RandomEmitterSet(10, 64)
        em.phot *= torch.rand((1,)) * 5000
        em.bg = 100 * torch.ones_like(em.bg)

        em_single = em.get_subset([0])
        assert tutil.tens_almeq(psf.crlb(em_single.xyz, em_single.phot, em_single.bg, 'xyzpb')[0],
                                psf.crlb_single(em.xyz, em.phot, em.bg, 'xyzpb')[0][0, :], 1e-4)

    def test_crlb_single(self, psf):
        em = emc.CoordinateOnlyEmitter(torch.tensor([[32., 32., 0.]]))
        em.phot = torch.tensor([5000.])
        em.bg = torch.tensor([10.])

        cr_m, img_m = psf.crlb(em.xyz, em.phot, em.bg)
        cr_s, img_s = psf.crlb_single(em.xyz, em.phot, em.bg)

        assert tutil.tens_almeq(cr_m, cr_s)
        assert tutil.tens_almeq(img_m, img_s)
        
        em = emc.RandomEmitterSet(20, 64)
        em.xyz[:, 2] *= (torch.rand_like(em.xyz[:, 2]) - 0.5) * 2 * 750
        em.phot *= 10000 * torch.rand_like(em.phot)
        em.bg = 10. * torch.ones_like(em.bg)

        cr_m, img_m = psf.crlb(em.xyz, em.phot, em.bg, crlb_order='xyzpb')
        cr_s, img_s = psf.crlb_single(em.xyz, em.phot, em.bg, crlb_order='xyzpb')

        ix_weird = (cr_m - cr_s < -1e-2)[:, :2].any(1)
        print(cr_m.sqrt() - cr_s.sqrt())
        em.phot *= 0
        em.phot[ix_weird] = 1.

        from deepsmlm.generic.plotting.frame_coord import PlotFrameCoord
        import matplotlib.pyplot as plt
        PlotFrameCoord(img_m, pos_tar=em.xyz, phot_tar=em.phot).plot()
        plt.colorbar()
        plt.show()

