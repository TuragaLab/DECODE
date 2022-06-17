import math
import pickle
from abc import ABC, abstractmethod

import numpy as np
import pytest
import torch

import decode.generic.test_utils as tutil
import decode.simulation.psf_kernel as psf_kernel

psf_cuda_available = pytest.mark.skipif(
    not psf_kernel.CubicSplinePSF.cuda_is_available(),
    reason="Skipped because cuda not available for Spline PSF.",
)


class AbstractPSFTest(ABC):
    @abstractmethod
    @pytest.fixture
    def psf(self):
        # put psf that outpus a 39 x 64 frame
        raise NotImplementedError

    @pytest.mark.parametrize(
        "ix_low,ix_high,ix_high_exp",
        [(None, None, 5), (-5, None, 8), (None, 3, 5), (2, 3, 1)],
    )
    def test_auto_filter_shift(self, ix_low, ix_high, ix_high_exp, psf):
        ix_low_exp = 0  # always 0 because it is shifted to there
        xyz, weight, frame_ix, ix_low, ix_high = psf._auto_filter_shift(
            torch.rand(5, 3), torch.rand(5), torch.arange(-2, 3), ix_low, ix_high
        )

        assert ix_low == ix_low_exp, "Wrong lower index"
        assert ix_high == ix_high_exp, "Wrong upper index"
        assert (
            frame_ix >= ix_low_exp
        ).all(), "Frame ix must not be lower than lower index"
        assert (frame_ix < ix_high_exp).all(), "Frame ix can not exceed upper index"

    @pytest.mark.parametrize("ix", ([(-5, 5), (0, 0), (0, 1), (0, 5)]))
    def test_forward_frame_dim(self, ix, psf):
        frames = psf.forward(torch.rand(42, 3), torch.ones(42), torch.arange(-21, 21), *ix)

        assert isinstance(frames, torch.Tensor), "Wrong output"
        assert len(frames) == ix[1] - ix[0], "Wrong output length"
        assert frames.size() == torch.Size([len(frames), 39, 64])

    def test_forward_frame_edge_cases(self, psf):
        """Tests whether the correct amount of frames are returned"""

        # No Emitters but indices
        out = psf.forward(
            torch.zeros((0, 3)), torch.ones(0), torch.zeros(0).long(), -5, 5
        )
        assert out.size() == torch.Size([10, 39, 64]), "Wrong frame dimensions"

        # Emitters but off indices
        out = psf.forward(
            torch.zeros((10, 3)), torch.ones(10), torch.zeros(10).long(), 5, 10
        )
        assert out.size() == torch.Size([5, 39, 64]), "Wrong frame dims"

        # Emitters and matching indices
        out = psf.forward(
            torch.zeros((10, 3)), torch.ones(10), torch.zeros(10).long(), -5, 5
        )
        assert out.size() == torch.Size([10, 39, 64]), "Wrong frame dims"

    def test_forward_frames_active(self, psf):
        """Tests whether the correct frames are active, i.e. signal present."""

        xyz = torch.Tensor([[15.0, 15.0, 0.0], [0.5, 0.3, 0.0]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.Tensor([0, 0]).int()

        frames = psf.forward(xyz, phot, frame_ix, -1, 2)

        assert frames.size() == torch.Size([3, 39, 64]), "Wrong frame dimensions."
        assert (frames[[0, -1]] == 0.0).all(), "Wrong frames active."
        assert (frames[1] != 0).any(), "Wrong frames active."
        assert frames[1].max() > 0, "Wrong frames active."


class TestPseudoPSF(AbstractPSFTest):
    @pytest.fixture()
    def psf(self):
        class QuasiAbstractPSF(psf_kernel.PSF):
            def _forward_single_frame(self, xyz: torch.Tensor, weight: torch.Tensor):
                # returns a frame where every pixel has the value of the number
                # of emitters on that frame
                return torch.ones(39, 64) * len(xyz)

        return QuasiAbstractPSF(None, None, None, (39, 64))

    def test_single_frame_wrapper(self, psf):
        """
        Tests whether the general impl
        Args:
            abs_psf:

        Returns:

        """

        xyz = torch.Tensor([[15.0, 15.0, 0.0], [0.5, 0.3, 0.0]])
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.Tensor([1, -2]).int()

        frames = psf._forward_single_frame_wrapper(xyz, phot, frame_ix, -2, 2)

        assert frames.size() == torch.Size([4, 39, 64])
        assert (frames[0] == 1).all()
        assert (frames[1:2] == 0).all()
        assert (frames[3] == 1).all()


class TestDeltaPSF(AbstractPSFTest):
    @pytest.fixture()
    def psf(self):
        return psf_kernel.DeltaPSF(
            xextent=(-0.5, 38.5), yextent=(-0.5, 63.5), img_shape=(39, 64)
        )

    @pytest.fixture()
    def delta_05px(self):
        return psf_kernel.DeltaPSF(
            xextent=(-0.5, 31.5), yextent=(-0.5, 31.5), img_shape=(64, 64)
        )

    def test_bin_ctr(self, psf, delta_05px):

        """Assert"""
        assert (psf._bin_ctr_x == torch.arange(39).float()).all()
        assert (psf._bin_ctr_y == torch.arange(64).float()).all()

        assert (delta_05px._bin_ctr_x == torch.arange(64).float() / 2 - 0.25).all()
        assert (delta_05px._bin_ctr_y == torch.arange(64).float() / 2 - 0.25).all()

    def test_px_search(self, delta_05px):
        # specific to 64px delta psf

        xyz = torch.tensor(
            [
                [-0.6, -0.5, 0.0],  # outside
                [-0.5, -0.5, 0.0],  # just inside
                [20.0, 30.0, 500.0],  # inside
                [31.4, 31.49, 0],  # just inside
                [31.5, 31.49, 0.0],  # just outside (open interval to the right)
                [50.0, 60.0, 0.0],
            ]
        )  # clearly outside

        xtar = torch.tensor([0, 41, 63])  # elmnts 1 2 3
        ytar = torch.tensor([0, 61, 63])

        with pytest.raises(ValueError):
            delta_05px.search_bin_index(xyz[:2])
        with pytest.raises(ValueError):
            delta_05px.search_bin_index(xyz[-3:-1])
        with pytest.raises(ValueError):
            delta_05px.search_bin_index(xyz[[-3, -1]])

        x, y = delta_05px.search_bin_index(xyz[1:4])

        assert isinstance(x, torch.LongTensor)
        assert isinstance(y, torch.LongTensor)
        assert (x == xtar).all()
        assert (y == ytar).all()

    def test_forward_manual(self, psf, delta_05px):
        # ToDo: Make this test prettier

        xyz = torch.Tensor(
            [
                [0.0, 0.0, 0.0],
                [15.0, 15.0, 200.0],
                [4.8, 4.8, 200.0],
                [4.79, 4.79, 500.0],
            ]
        )
        phot = torch.ones_like(xyz[:, 0])
        frame_ix = torch.tensor([1, 0, 1, 1])

        out_1px = psf.forward(xyz, phot, frame_ix, 0, 3)
        out_05px = delta_05px.forward(xyz, phot, frame_ix, 0, 3)

        assert out_1px.size() == torch.Size([3, 39, 64]), "Wrong output shape."
        assert out_05px.size() == torch.Size([3, 64, 64]), "Wrong output shape."

        assert out_1px[0, 15, 15] == 1.0
        assert out_05px[0, 31, 31] == 1.0

        assert out_1px[1, 0, 0] == 1.0
        assert out_05px[1, 1, 1] == 1.0

        assert (out_1px.unique() == torch.Tensor([0.0, 1.0])).all()
        assert (out_05px.unique() == torch.Tensor([0.0, 1.0])).all()

    def test_doubles(self, psf):
        # two emitters with different photon count in one pixel

        frames = psf.forward(
            torch.zeros((2, 3)),
            torch.tensor([1.0, 2.0]),
            torch.zeros(2).long(),
            0,
            1
        )

        # non-deterministic
        assert frames[0, 0, 0] in (1.0, 2.0)


class TestGaussianExpect(AbstractPSFTest):
    @pytest.fixture(scope="class", params=[None, (-5000.0, 5000.0)])
    def psf(self, request):
        return psf_kernel.GaussianPSF(
            (-0.5, 63.5), (-0.5, 63.5), request.param, img_shape=(39, 64), sigma_0=1.5
        )

    @pytest.mark.parametrize("norm", ["sum", "max"])
    def test_normalization(self, norm, psf):
        xyz = torch.tensor([[32.0, 32.0, 0.0]])
        phot = torch.tensor([1.0])

        psf.zextent = None  # measure in 2D
        psf.peak_weight = True if norm == "max" else False

        f = psf.forward(xyz, phot)

        if norm == "sum":
            assert f.sum().item() == pytest.approx(1, 0.05)
        elif norm == "max":
            assert f.max().item() == pytest.approx(1, 0.05)
        else:
            raise RuntimeError


class TestCubicSplinePSF(AbstractPSFTest):

    @pytest.fixture()
    def psf(self):
        xextent = (-0.5, 38.5)
        yextent = (-0.5, 63.5)
        img_shape = (39, 64)

        psf_impl = psf_kernel.CubicSplinePSF(
            xextent=xextent,
            yextent=yextent,
            img_shape=img_shape,
            ref0=(13, 13, 100),
            coeff=torch.rand(26, 26, 198, 64),
            vx_size=(1.0, 1.0, 10),
            roi_size=(32, 32),
            device="cpu",
        )

        return psf_impl

    @pytest.fixture()
    def psf_cuda(self, psf):
        # returns psf that lives on cuda
        return psf.cuda()

    @pytest.fixture()
    def onek_rois(self, psf):
        """Thousand random emitters in ROI"""
        n = 1000
        xyz = torch.rand((n, 3))
        xyz[:, :2] += psf.ref0[:2]
        xyz[:, 2] = xyz[:, 2] * 1000 - 500
        phot = torch.ones((n,)) * 10000
        bg = 50 * torch.ones((n,))

        return xyz, phot, bg, n

    def test_recentre_roi(self, psf):
        with pytest.raises(ValueError) as err:  # even roi size --> no center
            psf.__init__(
                xextent=psf.xextent,
                yextent=psf.yextent,
                img_shape=psf.img_shape,
                ref0=psf.ref0,
                coeff=psf._coeff,
                vx_size=psf.vx_size,
                roi_size=(24, 24),
                roi_auto_center=True,
                device="cpu",
            )

            assert err == "PSF reference can not be centered when the roi_size is even"

        with pytest.raises(ValueError) as err:  # even roi size --> no center
            psf.__init__(
                xextent=psf.xextent,
                yextent=psf.yextent,
                img_shape=psf.img_shape,
                ref0=psf.ref0,
                coeff=psf._coeff,
                vx_size=psf.vx_size,
                roi_size=(25, 25),
                ref_re=(5, 5, 100),
                roi_auto_center=True,
                device="cpu",
            )

            assert err == "PSF reference can not be automatically centered when you " \
                          "specify a custom center at the same time."

        psf.__init__(
            xextent=psf.xextent,
            yextent=psf.yextent,
            img_shape=psf.img_shape,
            ref0=psf.ref0,
            coeff=psf._coeff,
            vx_size=psf.vx_size,
            roi_size=(25, 25),
            roi_auto_center=True,
            device="cpu",
        )

        assert (psf.ref_re == torch.tensor([12, 12, psf.ref0[2]])).all()

    @psf_cuda_available
    def test_ship(self, psf, psf_cuda):
        import spline

        # test implementation before shipping to cuda
        assert isinstance(psf._spline_impl, spline.PSFWrapperCPU)
        assert isinstance(psf_cuda._spline_impl, spline.PSFWrapperCUDA)

        # test implementation state after cuda transition
        assert isinstance(psf.cuda()._spline_impl, spline.PSFWrapperCUDA)
        assert isinstance(psf_cuda.cpu()._spline_impl, spline.PSFWrapperCPU)

    def test_pickleability_cpu(self, psf):

        psf_str = pickle.dumps(psf)
        _ = pickle.loads(psf_str)

    @psf_cuda_available
    def test_pickleability_cuda(self, psf_cuda):

        self.test_pickleability_cpu(psf_cuda)

    @psf_cuda_available
    def test_roi_cuda_cpu(self, psf, psf_cuda, onek_rois):
        """Tests approximate equality of CUDA vs CPU implementation"""

        xyz, phot, bg, n = onek_rois
        phot = torch.ones_like(phot)

        roi_cpu = psf.forward_rois(xyz, phot)
        roi_cuda = psf_cuda.forward_rois(xyz, phot)

        assert tutil.tens_almeq(roi_cpu, roi_cuda, 1e-7)

    def test_roi_invariance(self, psf):
        """Tests whether shifts in x and y with multiples of px size lead to the same ROI"""

        # setup
        xyz_0 = torch.zeros((1, 3))

        steps = torch.arange(
            -5 * psf.vx_size[0], 5 * psf.vx_size[0], step=psf.vx_size[0]
        ).unsqueeze(1)

        # step coordinates in x and y direction
        xyz_x = torch.cat((steps, torch.zeros((steps.size(0), 2))), 1)
        xyz_y = torch.cat(
            (torch.zeros((steps.size(0), 1)), steps, torch.zeros((steps.size(0), 1))), 1
        )

        # run
        roi_ref = psf.forward_rois(xyz_0, torch.ones(1))
        roi_x = psf.forward_rois(xyz_x, torch.ones_like(xyz_x[:, 0]))
        roi_y = psf.forward_rois(xyz_y, torch.ones_like(xyz_y[:, 1]))

        # make sure that within a 5 x 5 window the values are the same
        assert tutil.tens_almeq(roi_ref[0, 10:15, 10:15], roi_x[:, 10:15, 10:15])
        assert tutil.tens_almeq(roi_ref[0, 10:15, 10:15], roi_y[:, 10:15, 10:15])

    @psf_cuda_available
    def test_roi_drv_cuda_cpu(self, psf, psf_cuda, onek_rois):
        """
        Tests approximate equality of CUDA and CPU implementation for a few
        ROIs on the derivatives.
        """
        xyz, phot, bg, n = onek_rois
        phot = torch.ones_like(phot)

        drv_roi_cpu, roi_cpu = psf.derivative(xyz, phot, bg)
        drv_roi_cuda, roi_cuda = psf_cuda.derivative(xyz, phot, bg)

        assert tutil.tens_almeq(drv_roi_cpu, drv_roi_cuda, 1e-7)
        assert tutil.tens_almeq(
            roi_cpu, roi_cuda, 1e-5
        )  # side output, seems to be a bit more numerially off

    def test_redefine_reference(self, psf):
        """Tests redefinition of reference point"""

        xyz = torch.tensor([[15.0, 15.0, 0.0]])
        roi_0 = psf.forward_rois(
            xyz,
            torch.ones(
                1,
            ),
        )

        # modify reference
        psf.__init__(
            xextent=psf.xextent,
            yextent=psf.yextent,
            img_shape=psf.img_shape,
            ref0=psf.ref0,
            coeff=psf._coeff,
            vx_size=psf.vx_size,
            roi_size=psf.roi_size_px,
            ref_re=psf.ref0 - torch.Tensor([1.0, 2.0, 0.0]),
            device=psf._device,
        )

        roi_shift = psf.forward_rois(
            xyz,
            torch.ones(
                1,
            ),
        )

        assert tutil.tens_almeq(roi_0[:, 5:10, 5:10], roi_shift[:, 4:9, 3:8])

    @psf_cuda_available
    def test_frame_cuda_cpu(self, psf, psf_cuda):
        """Tests approximate equality of CUDA vs CPU implementation"""
        n = 10000
        xyz = torch.rand((n, 3)) * 64
        xyz[:, 2] = torch.randn_like(xyz[:, 2]) * 1000 - 500
        phot = torch.ones((n,))
        frame_ix = torch.randint(0, 500, size=(n,))

        frames_cpu = psf.forward(xyz, phot, frame_ix)
        frames_cuda = psf_cuda.forward(xyz, phot, frame_ix)

        np.testing.assert_allclose(frames_cpu, frames_cuda, atol=1e-7)

    @pytest.mark.parametrize("ix_low,ix_high", [(0, 0), (-1, 1), (1, 1), (-5, 5)])
    def test_forward_chunks(self, psf, ix_low, ix_high):
        """Tests whether chunked forward returns the same frames as forward method"""

        n = 100
        xyz = torch.rand((n, 3)) * 64
        phot = torch.ones(n)
        frame_ix = torch.randint(-5, 4, size=(n,))

        out_chunk = psf._forward_chunks(
            xyz, phot, frame_ix, ix_low, ix_high, chunk_size=2
        )
        out_forward = psf.forward(xyz, phot, frame_ix, ix_low, ix_high)

        np.testing.assert_allclose(out_chunk, out_forward, rtol=1e-4)

    @pytest.mark.parametrize("ix_low,ix_high", [(0, 0), (-1, 1), (1, 1), (-5, 5)])
    def test_forward_drv_chunks(self, psf, ix_low, ix_high):
        """Tests whether chunked drv forward returns the same frames as drv forward method"""

        n = 100
        xyz = torch.rand((n, 3)) * 64
        phot = torch.ones(n)
        bg = torch.rand_like(phot) * 100

        drv_chunk, roi_chunk = psf._forward_drv_chunks(
            xyz, phot, bg, add_bg=False, chunk_size=2
        )
        drv, roi = psf.derivative(xyz, phot, bg, add_bg=False)

        np.testing.assert_allclose(drv_chunk, drv)
        np.testing.assert_allclose(roi_chunk, roi)

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
        psf_cuda.max_roi_chunk = 1000000
        n = psf_cuda._max_drv_roi_chunk * 5
        xyz = torch.rand((n, 3)) + 15
        phot = torch.ones(n)
        bg = torch.rand_like(phot) * 100

        drv, rois = psf_cuda.derivative(xyz, phot, bg)

        assert drv.size() == torch.Size([n, 5, *psf_cuda.roi_size_px])
        assert rois.size() == torch.Size([n, *psf_cuda.roi_size_px])

    def test_derivatives(self, psf, onek_rois):
        xyz, phot, bg, n = onek_rois

        drv, rois = psf.derivative(xyz, phot, bg)

        assert drv.size() == torch.Size(
            [n, psf.n_par, *psf.roi_size_px]
        ), "Wrong dimension of derivatives."
        np.testing.assert_allclose(
            drv[:, -1].unique(),
            torch.Tensor([0.0, 1.0]),
            err_msg="Derivative of background must be 1 or 0."
        )

        assert rois.size() == torch.Size(
            [n, *psf.roi_size_px]
        ), "Wrong dimension of ROIs."

    def test_fisher(self, psf, onek_rois):
        xyz, phot, bg, n = onek_rois

        fisher, rois = psf.fisher(xyz, phot, bg)

        assert fisher.size() == torch.Size([n, psf.n_par, psf.n_par])
        assert rois.size() == torch.Size(
            [n, *psf.roi_size_px]
        ), "Wrong dimension of ROIs."

    def test_crlb(self, psf, onek_rois):
        xyz, phot, bg, n = onek_rois
        alt_inv = torch.pinverse

        crlb, rois = psf.crlb(xyz, phot, bg)
        crlb_alt, rois_alt = psf.crlb(xyz, phot, bg, inversion=alt_inv)

        for cr, roi in zip((crlb, crlb_alt), (rois, rois_alt)):
            assert cr.size() == torch.Size([n, psf.n_par]), \
                "Wrong CRLB dimension."
            assert roi.size() == torch.Size([n, *psf.roi_size_px]), \
                "Wrong dimension of ROIs."


class TestZernikePSF(AbstractPSFTest):
    @pytest.fixture
    def psf(self):
        return psf_kernel.ZernikePSF(
            xextent=(-0.5, 38.5),
            yextent=(-0.5, 38.5),
            zextent=None,
            img_shape=(39, 64),
        )

    def test_phase_ramp(self, psf):
        ramp_x, ramp_y = psf._get_phase_ramp()

        for r in [ramp_x, ramp_y]:
            assert r.size() == torch.Size([39, 64])
            assert r.min() == 0
            assert r.max() == math.pi * 2
