import pathlib
import pickle
import random
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pytest
import torch

import decode.utils.calibration_io as load_cal
import decode.plot.frame_coord as plf
import decode.generic.test_utils as tutil
import decode.simulation.psf_kernel as psf_kernel
from . import asset_handler


psf_cuda_available = pytest.mark.skipif(not psf_kernel.CubicSplinePSF.cuda_is_available(), 
                                        reason="Skipped because cuda not available for Spline PSF.")


class AbstractPSFTest(ABC):

    @abstractmethod
    @pytest.fixture()
    def psf(self):
        raise NotImplementedError

    def test_forward_indices(self, psf):
        """Tests whether the correct amount of frames are returned"""

        # No Emitters but indices
        out = psf.forward(torch.zeros((0, 3)), torch.ones(0), torch.zeros(0).long(), -5, 5)
        assert out.size(0) == 11

        # Emitters but off indices
        out = psf.forward(torch.zeros((10, 3)), torch.ones(10), torch.zeros(10).long(), 5, 10)
        assert out.size(0) == 6

        # Emitters and matching indices
        out = psf.forward(torch.zeros((10, 3)), torch.ones(10), torch.zeros(10).long(), -5, 5)
        assert out.size(0) == 11

    def test_forward_frame_index(self, psf):
        """
        Tests whether the correct frames are active, i.e. signal present.

        Args:
            abs_psf: fixture

        """
        """Setup"""
        xyz = torch.Tensor([[15., 15., 0.], [0.5, 0.3, 0.]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.Tensor([0, 0]).int()

        """Run"""
        frames = psf.forward(xyz, phot, frame_ix, -1, 1)

        """Asserts"""
        assert frames.size() == torch.Size([3, 64, 64]), "Wrong dimensions."
        assert (frames[[0, -1]] == 0.).all(), "Wrong frames active."
        assert (frames[1] != 0).any(), "Wrong frames active."
        assert frames[1].max() > 0, "Wrong frames active."


class TestSingleFrameImplementedPSF(AbstractPSFTest):

    @pytest.fixture()
    def psf(self):
        """
        Abstract PSF class fixture

        Returns:
            psf_kernel.PSF: abstract PSF
        """

        class PseudoAbsPSF(psf_kernel.PSF):
            def _forward_single_frame(self, xyz: torch.Tensor, weight: torch.Tensor):
                if xyz.numel() >= 1:
                    return torch.ones((64, 64)) * xyz[0, 0]
                else:
                    return torch.zeros((64, 64))

            def forward(self, xyz: torch.Tensor, weight: torch.Tensor, frame_ix: torch.Tensor, ix_low, ix_high):
                xyz, weight, frame_ix, ix_low, ix_high = super().forward(xyz, weight, frame_ix, ix_low, ix_high)
                return self._forward_single_frame_wrapper(xyz, weight, frame_ix, ix_low, ix_high)

        return PseudoAbsPSF(None, None, None, None)

    def test_single_frame_wrapper(self, psf):
        """
        Tests whether the general impl
        Args:
            abs_psf:

        Returns:

        """

        xyz = torch.Tensor([[15., 15., 0.], [0.5, 0.3, 0.]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.Tensor([1, -2]).int()

        frames = psf._forward_single_frame_wrapper(xyz, phot, frame_ix, -2, 1)

        assert frames.dim() == 3
        assert (frames[0] == 0.5).all()
        assert (frames[1:3] == 0).all()
        assert (frames[-1] == 15).all()


class TestDeltaPSF:

    @pytest.fixture()
    def delta_1px(self):
        return psf_kernel.DeltaPSF(xextent=(-0.5, 31.5),
                                   yextent=(-0.5, 31.5),
                                   img_shape=(32, 32))

    @pytest.fixture()
    def delta_05px(self):
        return psf_kernel.DeltaPSF(xextent=(-0.5, 31.5),
                                   yextent=(-0.5, 31.5),
                                   img_shape=(64, 64))

    def test_bin_ctr(self, delta_1px, delta_05px):

        """Assert"""
        assert (delta_1px._bin_ctr_x == torch.arange(32).float()).all()
        assert (delta_1px._bin_ctr_y == torch.arange(32).float()).all()

        assert (delta_05px._bin_ctr_x == torch.arange(64).float() / 2 - 0.25).all()
        assert (delta_05px._bin_ctr_y == torch.arange(64).float() / 2 - 0.25).all()

    def test_px_search(self, delta_05px):

        """Setup"""
        xyz = torch.tensor([[-0.6, -0.5, 0.],  # outside
                            [-0.5, -0.5, 0.],  # just inside
                            [20., 30., 500.],  # inside
                            [31.4, 31.49, 0],  # just inside
                            [31.5, 31.49, 0.],  # just outside (open interval to the right)
                            [50., 60., 0.]])  # clearly outside

        xtar = torch.tensor([0, 41, 63])  # elmnts 1 2 3
        ytar = torch.tensor([0, 61, 63])

        """Run"""
        with pytest.raises(ValueError):
            delta_05px.search_bin_index(xyz[:2])
        with pytest.raises(ValueError):
            delta_05px.search_bin_index(xyz[-3:-1])
        with pytest.raises(ValueError):
            delta_05px.search_bin_index(xyz[[-3, -1]])

        x, y = delta_05px.search_bin_index(xyz[1:4])

        """Assert"""
        assert isinstance(x, torch.LongTensor)
        assert isinstance(y, torch.LongTensor)
        assert (x == xtar).all()
        assert (y == ytar).all()

    def test_forward(self, delta_1px, delta_05px):
        """
        Tests the implementation
        Args:
            delta_1px:
            delta_05px:

        Returns:

        """

        # ToDo: Not such a nice test ...

        xyz = torch.Tensor([[0., 0., 0.], [15.0, 15.0, 200.], [4.8, 4.8, 200.], [4.79, 4.79, 500.]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.tensor([1, 0, 1, 1])

        """Run"""
        out_1px = delta_1px.forward(xyz, phot, frame_ix, 0, 2)
        out_05px = delta_05px.forward(xyz, phot, frame_ix, 0, 2)

        """Test"""
        assert out_1px.size() == torch.Size([3, 32, 32]), "Wrong output shape."
        assert out_05px.size() == torch.Size([3, 64, 64]), "Wrong output shape."

        assert out_1px[0, 15, 15] == 1.
        assert out_05px[0, 31, 31] == 1.

        assert out_1px[1, 0, 0] == 1.
        assert out_05px[1, 1, 1] == 1.

        assert (out_1px.unique() == torch.Tensor([0., 1.])).all()
        assert (out_05px.unique() == torch.Tensor([0., 1.])).all()

    def test_forward_border(self, delta_1px):
        _ = delta_1px.forward(torch.zeros((0, 3)), torch.zeros((0,)), torch.zeros(0).long(), 0, 0)

    def test_doubles(self, delta_1px):
        frames = delta_1px.forward(torch.zeros((2, 3)), torch.tensor([1., 2.]), torch.zeros(2).long(), 0, 0)

        """Assert"""
        assert frames[0, 0, 0] in (1., 2.)


class TestGaussianExpect(TestSingleFrameImplementedPSF):

    @pytest.fixture(scope='class', params=[None, (-5000., 5000.)])
    def psf(self, request):
        return psf_kernel.GaussianPSF((-0.5, 63.5), (-0.5, 63.5), request.param, img_shape=(64, 64), sigma_0=1.5)

    def test_norm(self, psf):
        xyz = torch.tensor([[32., 32., 0.]])
        phot = torch.tensor([1.])
        assert pytest.approx(psf.forward(xyz, phot).sum().item(), 0.05) == 1

    def test_peak_weight(self, psf):
        psf.peak_weight = True

        xyz = torch.tensor([[32., 32., 0.]])
        phot = torch.tensor([1.])
        assert pytest.approx(psf.forward(xyz, phot).max().item(), 0.05) == 1

    def test_single_frame_wrapper(self, psf):
        """
        Tests whether the general impl
        Args:
            abs_psf:

        Returns:

        """

        xyz = torch.Tensor([[15., 15., 0.], [0.5, 0.3, 0.]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.Tensor([1, -2]).int()

        frames = psf._forward_single_frame_wrapper(xyz, phot, frame_ix, -2, 1)

        assert frames.dim() == 3
        assert (frames[0] != 0).any()
        assert (frames[1:3] == 0).all()
        assert (frames[-1] != 0).any()


class TestCubicSplinePSF(AbstractPSFTest):
    cdir = pathlib.Path(__file__).resolve().parent
    bead_cal_file = (cdir / pathlib.Path('assets/bead_cal_for_testing_3dcal.mat'))  # expected path, might not exist

    @pytest.fixture()
    def psf(self):
        xextent = (-0.5, 63.5)
        yextent = (-0.5, 63.5)
        img_shape = (64, 64)

        """Have a look whether the bead calibration is there"""
        asset_handler.AssetHandler().auto_load(self.bead_cal_file)

        smap_psf = load_cal.SMAPSplineCoefficient(calib_file=str(self.bead_cal_file))
        psf_impl = psf_kernel.CubicSplinePSF(xextent=xextent, yextent=yextent, img_shape=img_shape, ref0=smap_psf.ref0,
                                             coeff=smap_psf.coeff, vx_size=(1., 1., 10), roi_size=(32, 32),
                                             device='cpu')

        return psf_impl

    @pytest.fixture()
    def psf_cuda(self, psf):
        """
        Returns CUDA version of CPU implementation. Will make tests fail if not compiled with CUDA support enabled.
        Args:
            psf:

        Returns:

        """
        return psf.cuda()

    @pytest.fixture()
    def onek_rois(self, psf):
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
        xyz[:, :2] += psf.ref0[:2]
        xyz[:, 2] = xyz[:, 2] * 1000 - 500
        phot = torch.ones((n,)) * 10000
        bg = 50 * torch.ones((n,))

        return xyz, phot, bg, n

    def test_recentre_roi(self, psf):
        with pytest.raises(ValueError) as err:  # even roi size --> no center
            psf.__init__(xextent=psf.xextent, yextent=psf.yextent, img_shape=psf.img_shape, ref0=psf.ref0,
                         coeff=psf._coeff, vx_size=psf.vx_size, roi_size=(24, 24), roi_auto_center=True,
                         device='cpu')

            assert err == 'PSF reference can not be centered when the roi_size is even.'

        with pytest.raises(ValueError) as err:  # even roi size --> no center
            psf.__init__(xextent=psf.xextent, yextent=psf.yextent, img_shape=psf.img_shape, ref0=psf.ref0,
                         coeff=psf._coeff, vx_size=psf.vx_size, roi_size=(25, 25), ref_re=(5, 5, 100),
                         roi_auto_center=True, device='cpu')

            assert err == 'PSF reference can not be automatically centered when you specify a custom center at the same time.'

        psf.__init__(xextent=psf.xextent, yextent=psf.yextent, img_shape=psf.img_shape, ref0=psf.ref0,
                     coeff=psf._coeff, vx_size=psf.vx_size, roi_size=(25, 25), roi_auto_center=True,
                     device='cpu')

        assert (psf.ref_re == torch.tensor([12, 12, psf.ref0[2]])).all()

    def test_forward_ix_out_of_range(self, psf):
        out = psf.forward(torch.zeros((0, 3)), torch.zeros(0), torch.zeros(0).long(), -5, 5)
        assert out.size(0) == 11

    @psf_cuda_available
    def test_ship(self, psf, psf_cuda):
        """
        Tests ships to CPU / CUDA
        Args:
            psf:
            psf_cuda:

        Returns:

        """
        import spline

        """Test implementation state before"""
        assert isinstance(psf._spline_impl, spline.PSFWrapperCPU)
        assert isinstance(psf_cuda._spline_impl, spline.PSFWrapperCUDA)

        """Test implementation state after"""
        assert isinstance(psf.cuda()._spline_impl, spline.PSFWrapperCUDA)
        assert isinstance(psf_cuda.cpu()._spline_impl, spline.PSFWrapperCPU)

    def test_pickleability_cpu(self, psf):
        """
        Tests the pickability of CPU implementation

        Args:
            psf: fixture

        """

        psf_str = pickle.dumps(psf)
        _ = pickle.loads(psf_str)

    @psf_cuda_available
    def test_pickleability_cuda(self, psf_cuda):
        self.test_pickleability_cpu(psf_cuda)

    @psf_cuda_available
    def test_roi_cuda_cpu(self, psf, psf_cuda, onek_rois):
        """
        Tests approximate equality of CUDA vs CPU implementation for a few ROIs

        Args:
            psf: psf implementation on CPU
            psf_cuda: psf implementation on CUDA

        """
        xyz, phot, bg, n = onek_rois
        phot = torch.ones_like(phot)

        roi_cpu = psf.forward_rois(xyz, phot)
        roi_cuda = psf_cuda.forward_rois(xyz, phot)

        assert tutil.tens_almeq(roi_cpu, roi_cuda, 1e-7)

    def test_roi_invariance(self, psf):
        """
        Tests whether shifts in x and y with multiples of px size lead to the same ROI

        Args:
            psf: fixture

        """
        """Setup"""
        xyz_0 = torch.zeros((1, 3))

        steps = torch.arange(-5 * psf.vx_size[0], 5 * psf.vx_size[0], step=psf.vx_size[0]).unsqueeze(1)

        # step coordinates in x and y direction
        xyz_x = torch.cat((steps, torch.zeros((steps.size(0), 2))), 1)
        xyz_y = torch.cat((torch.zeros((steps.size(0), 1)), steps, torch.zeros((steps.size(0), 1))), 1)

        """Run"""
        roi_ref = psf.forward_rois(xyz_0, torch.ones(1))
        roi_x = psf.forward_rois(xyz_x, torch.ones_like(xyz_x[:, 0]))
        roi_y = psf.forward_rois(xyz_y, torch.ones_like(xyz_y[:, 1]))

        """Assertions"""
        # make sure that within a 5 x 5 window the values are the same

        assert tutil.tens_almeq(roi_ref[0, 10:15, 10:15], roi_x[:, 10:15, 10:15])
        assert tutil.tens_almeq(roi_ref[0, 10:15, 10:15], roi_y[:, 10:15, 10:15])

    @pytest.mark.plot
    def test_roi_visual(self, psf, onek_rois):
        xyz, phot, bg, n = onek_rois
        phot = torch.ones_like(phot)
        roi_cpu = psf.forward_rois(xyz, phot)

        """Additional Plotting if manual testing (comment out return statement)"""
        rix = random.randint(0, n - 1)
        plt.figure()
        plf.PlotFrameCoord(roi_cpu[rix], pos_tar=xyz[[rix]]).plot()
        plt.title(f"Random ROI sample.\nShould show a single emitter it the reference point of the psf.\n"
                  f"Reference: {psf.ref0}")
        plt.show()

    @psf_cuda_available
    def test_roi_drv_cuda_cpu(self, psf, psf_cuda, onek_rois):
        """
        Tests approximate equality of CUDA and CPU implementation for a few ROIs on the derivatives.

        Args:
            psf: psf implementation on CPU
            psf_cuda: psf implementation on CUDA

        """
        xyz, phot, bg, n = onek_rois
        phot = torch.ones_like(phot)

        drv_roi_cpu, roi_cpu = psf.derivative(xyz, phot, bg)
        drv_roi_cuda, roi_cuda = psf_cuda.derivative(xyz, phot, bg)

        assert tutil.tens_almeq(drv_roi_cpu, drv_roi_cuda, 1e-7)
        assert tutil.tens_almeq(roi_cpu, roi_cuda, 1e-5)  # side output, seems to be a bit more numerially off

    @pytest.mark.plot
    def test_roi_drv_visual(self, psf, onek_rois):
        xyz, phot, bg, n = onek_rois
        phot = torch.ones_like(phot)

        drv_rois, rois = psf.derivative(xyz, phot, bg)

        """Additional Plotting if manual testing."""
        rix = random.randint(0, n - 1)  # pick random sample
        dr = drv_rois[rix]
        r = rois[rix]
        xyzr = xyz[[rix]]

        plt.figure(figsize=(20, 12))

        plt.subplot(231)
        plf.PlotFrameCoord(r, pos_tar=xyzr).plot()
        plt.title(f"Random ROI sample.\nShould show a single emitter it the reference point of the psf.\n"
                  f"Reference: {psf.ref0}")

        plt.subplot(232)
        plf.PlotFrame(dr[0], plot_colorbar=True).plot()
        plt.title('d/dx')

        plt.subplot(233)
        plf.PlotFrame(dr[1], plot_colorbar=True).plot()
        plt.title('d/dy')

        plt.subplot(234)
        plf.PlotFrame(dr[2], plot_colorbar=True).plot()
        plt.title('d/dz')

        plt.subplot(235)
        plf.PlotFrame(dr[3], plot_colorbar=True).plot()
        plt.title('d/dphot')

        plt.subplot(236)
        plf.PlotFrame(dr[4], plot_colorbar=True).plot()
        plt.title('d/dbg')

        plt.show()

    def test_redefine_reference(self, psf):
        """
        Tests redefinition of reference

        Args:
            psf: fixture

        """

        """Assert and test"""
        xyz = torch.tensor([[15., 15., 0.]])
        roi_0 = psf.forward_rois(xyz, torch.ones(1, ))

        # modify reference
        psf.__init__(xextent=psf.xextent, yextent=psf.yextent, img_shape=psf.img_shape,
                     ref0=psf.ref0, coeff=psf._coeff, vx_size=psf.vx_size,
                     roi_size=psf.roi_size_px, ref_re=psf.ref0 - torch.Tensor([1., 2., 0.]),
                     device=psf._device)

        roi_shift = psf.forward_rois(xyz, torch.ones(1, ))

        assert tutil.tens_almeq(roi_0[:, 5:10, 5:10], roi_shift[:, 4:9, 3:8])

    @psf_cuda_available
    def test_frame_cuda_cpu(self, psf, psf_cuda):
        """
        Tests approximate equality of CUDA vs CPU implementation for a few frames

        Args:
            psf: psf fixture, CPU version
            psf_cuda: psf fixture, CUDA version

        Returns:

        """
        n = 10000
        xyz = torch.rand((n, 3)) * 64
        xyz[:, 2] = torch.randn_like(xyz[:, 2]) * 1000 - 500
        phot = torch.ones((n,))
        frame_ix = torch.randint(0, 500, size=(n,))

        frames_cpu = psf.forward(xyz, phot, frame_ix)
        frames_cuda = psf_cuda.forward(xyz, phot, frame_ix)

        assert tutil.tens_almeq(frames_cpu, frames_cuda, 1e-7)

    @pytest.mark.plot
    def test_frame_visual(self, psf):
        n = 10
        xyz = torch.rand((n, 3)) * 64
        xyz[:, 2] = torch.randn_like(xyz[:, 2]) * 1000 - 500
        phot = torch.ones((n,))
        frame_ix = torch.zeros_like(phot).int()

        frames_cpu = psf.forward(xyz, phot, frame_ix)

        """Additional Plotting if manual testing (comment out return statement)."""
        plt.figure()
        plf.PlotFrameCoord(frames_cpu[0], pos_tar=xyz).plot()
        plt.title(
            "Random Frame sample.\nShould show a couple of emitters at\nrandom positions distributed over a frame.")
        plt.show()

    @pytest.mark.parametrize("ix_low,ix_high", [(0, 0), (-1, 1), (1, 1), (-5, 5)])
    def test_forward_chunks(self, psf, ix_low, ix_high):
        """
        Tests whether chunked forward returns the same frames as forward method

        Args:
            psf: fixture

        """

        """Setup"""
        n = 100
        xyz = torch.rand((n, 3)) * 64
        phot = torch.ones(n)
        frame_ix = torch.randint(-5, 4, size=(n,))

        """Run"""
        out_chunk = psf._forward_chunks(xyz, phot, frame_ix, ix_low, ix_high, chunk_size=2)
        out_forward = psf.forward(xyz, phot, frame_ix, ix_low, ix_high)

        """Test"""
        assert tutil.tens_almeq(out_chunk, out_forward)

    @pytest.mark.parametrize("ix_low,ix_high", [(0, 0), (-1, 1), (1, 1), (-5, 5)])
    def test_forward_drv_chunks(self, psf, ix_low, ix_high):
        """
        Tests whether chunked drv forward returns the same frames as drv forward method

        Args:
           psf: fixture

        """

        """Setup"""
        n = 100
        xyz = torch.rand((n, 3)) * 64
        phot = torch.ones(n)
        bg = torch.rand_like(phot) * 100

        """Run"""
        drv_chunk, roi_chunk = psf._forward_drv_chunks(xyz, phot, bg, add_bg=False, chunk_size=2)
        drv, roi = psf.derivative(xyz, phot, bg, add_bg=False)

        """Test"""
        assert tutil.tens_almeq(drv_chunk, drv)
        assert tutil.tens_almeq(roi_chunk, roi)

    @pytest.mark.slow
    @psf_cuda_available
    def test_many_em_forward(self, psf_cuda):
        """Setup"""
        psf_cuda.max_roi_chunk = 1000000
        n = psf_cuda.max_roi_chunk * 5
        n_frames = n // 50
        xyz = torch.rand((n, 3)) + 15
        phot = torch.ones(n)
        frame_ix = torch.randint(0, n_frames, size=(n,)).long()

        """Run"""
        frames = psf_cuda.forward(xyz, phot, frame_ix, 0, n_frames)

        """Assert"""
        assert frames.size() == torch.Size([n_frames + 1, 64, 64])

    @pytest.mark.slow
    @psf_cuda_available
    def test_many_drv_roi_forward(self, psf_cuda):
        """Setup"""
        psf_cuda.max_roi_chunk = 1000000
        n = psf_cuda._max_drv_roi_chunk * 5
        xyz = torch.rand((n, 3)) + 15
        phot = torch.ones(n)
        bg = torch.rand_like(phot) * 100

        """Run"""
        drv, rois = psf_cuda.derivative(xyz, phot, bg)

        """Assert"""
        assert drv.size() == torch.Size([n, 5, *psf_cuda.roi_size_px])
        assert rois.size() == torch.Size([n, *psf_cuda.roi_size_px])

    def test_derivatives(self, psf, onek_rois):
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
        drv, rois = psf.derivative(xyz, phot, bg)

        """Test"""
        assert drv.size() == torch.Size([n, psf.n_par, *psf.roi_size_px]), "Wrong dimension of derivatives."
        assert tutil.tens_almeq(drv[:, -1].unique(), torch.Tensor([0., 1.])), "Derivative of background must be 1 or 0."

        assert rois.size() == torch.Size([n, *psf.roi_size_px]), "Wrong dimension of ROIs."

    def test_fisher(self, psf, onek_rois):
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
        fisher, rois = psf.fisher(xyz, phot, bg)

        """Test"""
        assert fisher.size() == torch.Size([n, psf.n_par, psf.n_par])

        assert rois.size() == torch.Size([n, *psf.roi_size_px]), "Wrong dimension of ROIs."

    @pytest.mark.xfail(float(torch.__version__[:3]) < 1.4,
                       reason="Pseudo inverse is not implemented in batch mode for older pytorch versions.")
    def test_crlb(self, psf, onek_rois):
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
        crlb, rois = psf.crlb(xyz, phot, bg)
        crlb_p, _ = psf.crlb(xyz, phot, bg, inversion=alt_inv)

        """Test"""
        assert crlb.size() == torch.Size([n, psf.n_par]), "Wrong CRLB dimension."
        assert (torch.Tensor([.01, .01, .02]) ** 2 <= crlb[:, :3]).all(), "CRLB in wrong range (lower bound)."
        assert (torch.Tensor([.1, .1, 100]) ** 2 >= crlb[:, :3]).all(), "CRLB in wrong range (upper bound)."

        diff_inv = (crlb_p - crlb).abs()

        assert tutil.tens_almeq(diff_inv[:, :2], torch.zeros_like(diff_inv[:, :2]), 1e-4)
        assert tutil.tens_almeq(diff_inv[:, 2], torch.zeros_like(diff_inv[:, 2]), 1e-1)
        assert tutil.tens_almeq(diff_inv[:, 3], torch.zeros_like(diff_inv[:, 3]), 1e2)
        assert tutil.tens_almeq(diff_inv[:, 4], torch.zeros_like(diff_inv[:, 4]), 1e-3)

        assert rois.size() == torch.Size([n, *psf.roi_size_px]), "Wrong dimension of ROIs."
