import random

import matplotlib.pyplot as plt
import pytest
import torch

import deepsmlm.generic.inout.load_calibration as load_cal
import deepsmlm.generic.plotting.frame_coord as plf
import deepsmlm.generic.psf_kernel as psf_kernel
import deepsmlm.generic.utils.test_utils as tutil


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

    @pytest.fixture(scope='class')
    def onek_rois(self, psf_cpu):
        """
        Thousand random emitters in ROI

        Returns:
            xyz:
            phot:
            bg:
            n (int): number of emitters

        """
        n = 1000
        xyz = torch.rand((n, 3))
        xyz[:, :2] += psf_cpu.ref0[:2]
        xyz[:, 2] = xyz[:, 2] * 1000 - 500
        phot = torch.ones((n,)) * 10000
        bg = 50 * torch.ones((n,))

        return xyz, phot, bg, n

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
        phot = torch.ones((n,))

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
            psf_cpu: psf fixture, CPU version
            psf_cuda: psf fixture, CUDA version

        Returns:

        """
        n = 10000
        xyz = torch.rand((n, 3)) * 64
        xyz[:, 2] = torch.randn_like(xyz[:, 2]) * 1000 - 500
        phot = torch.ones((n,))
        frame_ix = torch.randint(0, 500, size=(n,))

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

    def test_derivatives(self, psf_cuda, onek_rois):
        """
        Tests the derivate calculation

        Args:
            psf_cuda: psf fixture (see above)
            onek_rois: 1k roi fixture (see above)

        Returns:

        """
        """Setup"""
        xyz, phot, bg, n = onek_rois

        """Run"""
        drv, rois = psf_cuda.derivative(xyz, phot, bg)

        """Test"""
        assert drv.size() == torch.Size([n, psf_cuda.n_par, *psf_cuda.roi_size_px]), "Wrong dimension of derivatives."
        assert tutil.tens_almeq(drv[:, -1].unique(), torch.Tensor([0., 1.])), "Derivative of background must be 1 or 0."

        assert rois.size() == torch.Size([n, *psf_cuda.roi_size_px]), "Wrong dimension of ROIs."

    def test_fisher(self, psf_cuda, onek_rois):
        """
        Tests the fisher matrix calculation.

        Args:
            psf_cuda: psf fixture (see above)
            onek_rois: 1k roi fixture (see above)

        Returns:

        """
        """Setup"""
        xyz, phot, bg, n = onek_rois

        """Run"""
        fisher, rois = psf_cuda.fisher(xyz, phot, bg)

        """Test"""
        assert fisher.size() == torch.Size([n, psf_cuda.n_par, psf_cuda.n_par])

        assert rois.size() == torch.Size([n, *psf_cuda.roi_size_px]), "Wrong dimension of ROIs."

    def test_crlb(self, psf_cuda, onek_rois):
        """
        Tests the crlb calculation
        Args:
            psf_cuda: psf fixture (see above)
            onek_rois: 1k roi fixture (see above)

        Returns:

        """
        """Setup"""
        xyz, phot, bg, n = onek_rois
        alt_inv = torch.pinverse

        """Run"""
        crlb, rois = psf_cuda.crlb(xyz, phot, bg)
        if float(torch.__version__[:3]) >= 1.4:
            crlb_p, _ = psf_cuda.crlb(xyz, phot, bg, inversion=alt_inv)

        """Test"""
        assert crlb.size() == torch.Size([n, psf_cuda.n_par]), "Wrong CRLB dimension."
        assert (torch.Tensor([.01, .01, .02]) ** 2 <= crlb[:, :3]).all(), "CRLB in wrong range (lower bound)."
        assert (torch.Tensor([.1, .1, 100]) ** 2 >= crlb[:, :3]).all(), "CRLB in wrong range (upper bound)."

        if float(torch.__version__[:3]) >= 1.4:
            assert tutil.tens_almeq(crlb, crlb_p, 1e-1)

        assert rois.size() == torch.Size([n, *psf_cuda.roi_size_px]), "Wrong dimension of ROIs."
