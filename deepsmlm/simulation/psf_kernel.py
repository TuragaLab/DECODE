import math
import numpy as np
import warnings
from abc import ABC, abstractmethod  # abstract class
import torch
from scipy.stats import binned_statistic_2d

from deepsmlm.generic.utils import generic as gutil
import spline_psf_cuda  # CPP / CUDA implementation


class PSF(ABC):
    """
    Abstract class to represent a point spread function.
    forward must be overwritten and shall be called via super().forward(...) before the subclass implementation follows.
    """

    def __init__(self, xextent=(None, None), yextent=(None, None), zextent=None, img_shape=(None, None)):
        """
        Constructor to comprise a couple of default attributes

        Args:
            xextent:
            yextent:
            zextent:
            img_shape:
        """

        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.img_shape = img_shape

    def __str__(self):
        return 'PSF: \n xextent: {}\n yextent: {}\n zextent: {}\n img_shape: {}'.format(self.xextent,
                                                                                       self.yextent,
                                                                                       self.zextent,
                                                                                       self.img_shape)

    def crlb(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, xyz: torch.Tensor, weight: torch.Tensor, frame_ix: torch.Tensor, ix_low: int, ix_high: int):
        """
        Forward coordinates frame index aware through the psf model.
        Implementation methods should call this method first in order not to handle the default argument stuff.

        Example::
            If implementation do not implement a batched forward method, they may call this method, and then refer to the
            single-frame wrapper as. Their forward implementation will then look like this:

            >>> xyz, weight, frame_ix, ix_low, ix_high = super().forward(xyz, weight, frame_ix, ix_low, ix_high)
            >>> return self._forward_single_frame_wrapper(xyz, weight, frame_ix, ix_low, ix_high)

        Args:
            xyz: coordinates of size N x (2 or 3)
            weight: photon values of size N or None
            frame_ix: (optional) frame index
            ix_low: (optional) lower frame_index, if None will be determined automatically
            ix_high: (optional) upper frame_index, if None will be determined automatically

        Returns:
            frames (torch.Tensor): frames of size N x H x W where N is the batch dimension.
        """
        if frame_ix is None:
            frame_ix = torch.zeros((xyz.size(0), )).int()

        if ix_low is None:
            ix_low = frame_ix.min().item()

        if ix_high is None:
            ix_high = frame_ix.max().item()

        """Kick out everything that is out of frame index bounds and shift the frames to start at 0"""
        in_frame = (ix_low <= frame_ix) * (frame_ix <= ix_high)
        xyz_ = xyz[in_frame, :]
        weight_ = weight[in_frame] if weight is not None else None
        frame_ix_ = frame_ix[in_frame] - ix_low

        ix_high = ix_high - ix_low
        ix_low = 0

        return xyz_, weight_, frame_ix_, ix_low, ix_high

    def _forward_single_frame(self, xyz: torch.Tensor, weight: torch.Tensor):
        raise NotImplementedError

    def _forward_single_frame_wrapper(self, xyz: torch.Tensor, weight: torch.Tensor, frame_ix: torch.Tensor,
                                      ix_low: int, ix_high: int):
        """
        This is a convenience (fallback) wrapper that splits the input in frames and forward them throguh the single
        frame
        function if the implementation does not have a frame index (i.e. batched) forward method.

        Args:
            xyz: coordinates of size N x (2 or 3)
            weight: photon value
            frame_ix: frame index
            ix_low (int):  lower frame_index, if None will be determined automatically
            ix_high (int):  upper frame_index, if None will be determined automatically

        Returns:
            frames (torch.Tensor): N x H x W, stacked frames
        """
        ix_split, n_splits = gutil.ix_split(frame_ix, ix_min=ix_low, ix_max=ix_high)
        if weight is not None:
            frames = [self._forward_single_frame(xyz[ix_split[i]], weight[ix_split[i]]) for i in range(n_splits)]
        else:
            frames = [self._forward_single_frame(xyz[ix_split[i]], None) for i in range(n_splits)]
        frames = torch.stack(frames, 0)

        return frames


class DeltaPSF(PSF):
    """
    Delta function PSF. You input a list of coordinates, this class outputs a single one-hot representation in 2D of
    your input.
    If multiple things fall into the same bin, the output is the weight of either of the two (which one is arbitrary
    implementation detail).

    """

    def __init__(self, xextent, yextent, img_shape):
        super().__init__(xextent=xextent, yextent=yextent, img_shape=img_shape)

        from deepsmlm.neuralfitter.pre_processing import RemoveOutOfFOV

        self._fov_filter = RemoveOutOfFOV(xextent=self.xextent, yextent=self.yextent, zextent=None)
        self._x_bins = torch.linspace(*xextent, steps=img_shape[0] + 1)
        self._y_bins = torch.linspace(*yextent, steps=img_shape[1] + 1)

    def _delta_impl(self, xyz: torch.Tensor, weight: torch.Tensor, frame_ix: torch.LongTensor, n_frames: int):
        """
        Implementation via searchsorted.

        Args:
            xyz:
            weight:
            frame_ix:
            n_frames:

        Returns:

        """

        """Remove Emitters that are out of FOV"""
        mask = self._fov_filter.clean_emitter(xyz)
        assert isinstance(frame_ix, (torch.IntTensor, torch.LongTensor, torch.ShortTensor))
        xyz_, weight_, frame_ix_ = xyz[mask], weight[mask], frame_ix[mask].long()

        """Generate frames"""
        frames = torch.zeros((n_frames, *self.img_shape))

        x_ix = np.searchsorted(self._x_bins, xyz_[:, 0], side='right') - 1
        y_ix = np.searchsorted(self._y_bins, xyz_[:, 1], side='right') - 1

        frames[frame_ix_, x_ix, y_ix] = weight_

        return frames

    def forward(self, xyz: torch.Tensor, weight: torch.Tensor = None, frame_ix: torch.Tensor = None,
                ix_low=None, ix_high=None):
        """
        Forward coordinates frame index aware through the psf model.

        Args:
            xyz: coordinates of size N x (2 or 3)
            weight: photon value
            frame_ix: (optional) frame index
            ix_low: (optional) lower frame_index, if None will be determined automatically
            ix_high: (optional) upper frame_index, if None will be determined automatically

        Returns:
            frames (torch.Tensor): frames of size N x H x W where N is the batch dimension.
        """

        xyz, weight, frame_ix, ix_low, ix_high = super().forward(xyz, weight, frame_ix, ix_low, ix_high)
        return self._delta_impl(xyz, weight, frame_ix, ix_high - ix_low + 1)


class GaussianExpect(PSF):
    """
    A gaussian PSF which models the by using the function values of the probability distribution.
    You must not use this function without shotnoise for simulation.
    """

    def __init__(self, xextent, yextent, zextent, img_shape, sigma_0, peak_weight=False):
        """
        Init of Gaussian Expect. If no z extent is provided we assume 2D PSF.

        Args:
            xextent: (tuple of float) extent of psf in x
            yextent: (tuple of float) extent of psf in y
            zextent: (tuple of float or None, optional) extent of psf in z
            img_shape: (tuple) img shape
            sigma_0: sigma in focus
            peak_weight: (bool) if true: use peak intensity instead of integral under the curve
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape)

        self.sigma_0 = sigma_0
        self.peak_weight = peak_weight

    @staticmethod
    def astigmatism(z, sigma_0=1.00, foc_shift=250, rl_range=280.0):
        """
        Computes width of gaussian dependent on the z value as by the Gaussian Beam model.

        Args:
            z: (torch.Tensor, N) z values
            sigma_0: initial sigma in px units
            foc_shift: focal shift upon introduction of cylindrical lens in nm
            rl_range: rayleigh range in nm

        Returns:
            sigma values for x and y
        """

        sigma_x = sigma_0 * (1 + ((z + foc_shift)/(rl_range))**2).sqrt()
        sigma_y = sigma_0 * (1 + ((z - foc_shift)/(rl_range))**2).sqrt()

        sigma_xy = torch.cat((sigma_x.unsqueeze(1), sigma_y.unsqueeze(1)), 1)

        return sigma_xy

    def _forward_single_frame(self, xyz: torch.Tensor, weight: torch.Tensor):
        """
        Calculates the PSF for emitters on the same frame

        Args:
            xyz: coordinates
            weight: weight

        Returns:
            (torch.Tensor) size H x W

        """
        num_emitters = xyz.shape[0]
        img_shape = self.img_shape
        sigma_0 = self.sigma_0

        if num_emitters == 0:
            return torch.zeros(img_shape[0], img_shape[1]).float()

        xpos = xyz[:, 0].repeat(img_shape[0], img_shape[1], 1)
        ypos = xyz[:, 1].repeat(img_shape[0], img_shape[1], 1)

        if self.zextent is not None:
            sig = self.astigmatism(xyz[:, 2], sigma_0=self.sigma_0)
            sig_x = sig[:, 0].repeat(img_shape[0], img_shape[1], 1)
            sig_y = sig[:, 1].repeat(img_shape[0], img_shape[1], 1)
        else:
            sig_x = sigma_0
            sig_y = sigma_0

        x = torch.linspace(self.xextent[0], self.xextent[1], img_shape[0] + 1).float()
        y = torch.linspace(self.yextent[0], self.yextent[1], img_shape[1] + 1).float()
        xx, yy = torch.meshgrid(x, y)

        xx = xx.unsqueeze(2).repeat(1, 1, num_emitters)
        yy = yy.unsqueeze(2).repeat(1, 1, num_emitters)

        # print(xx.shape)

        gauss_x = torch.erf((xx[1:, 1:, :] - xpos) / (math.sqrt(2) * sig_x)) \
            - torch.erf((xx[0:-1, 1:, :] - xpos) / (math.sqrt(2) * sig_x))

        gauss_y = torch.erf((yy[1:, 1:, :] - ypos) / (math.sqrt(2) * sig_y)) \
            - torch.erf((yy[1:, 0:-1, :] - ypos) / (math.sqrt(2) * sig_y))

        gaussCdf = weight.type_as(gauss_x) / 4 * torch.mul(gauss_x, gauss_y)
        if self.peak_weight:
            gaussCdf *= 2 * math.pi * sig_x * sig_y
        gaussCdf = torch.sum(gaussCdf, 2)

        return gaussCdf

    def forward(self, xyz: torch.Tensor, weight: torch.Tensor, frame_ix: torch.Tensor = None, ix_low=None,
                ix_high=None):
        """
        Forward coordinates frame index aware through the psf model.

        Args:
            xyz: coordinates of size N x (2 or 3)
            weight: photon value
            frame_ix: (optional) frame index
            ix_low: (optional) lower frame_index, if None will be determined automatically
            ix_high: (optional) upper frame_index, if None will be determined automatically

        Returns:
            frames (torch.Tensor): frames of size N x H x W where N is the batch dimension.
        """
        xyz, weight, frame_ix, ix_low, ix_high = super().forward(xyz, weight, frame_ix, ix_low, ix_high)
        return self._forward_single_frame_wrapper(xyz=xyz, weight=weight, frame_ix=frame_ix,
                                                  ix_low=ix_low, ix_high=ix_high)


class CubicSplinePSF(PSF):
    """
    Cubic spline PSF.
    """

    n_par = 5  # x, y, z, phot, bg
    inv_default = torch.inverse

    def __init__(self, xextent, yextent, img_shape, ref0, coeff, vx_size, *, roi_size=None, ref_re=None,
                 cuda_kernel=True, cuda_max_roi_chunk: int = 1000000):
        """
        Initialise Spline PSF

        Args:
            ref_re:
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape (tuple): img_shape
            coeff: spline coefficients
            ref0 (tuple): zero reference point in implementation units
            vx_size (tuple): pixel / voxel size
            roi_size (tuple, None, optional): roi_size. optional. can be determined from dimension of coefficients.
            cuda_kernel: use cuda implementation
            cuda_max_roi_chunk (int): max number of rois to be processed at a time via the cuda kernel. If you run into
                memory allocation errors, decrease this number or free some space on your CUDA device.
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=None, img_shape=img_shape)

        self._coeff = coeff
        self._roi_native = self._coeff.size()[:2]  # native roi based on the coeff's size
        self.roi_size_px = roi_size if roi_size is not None else self._roi_native
        if vx_size is None:
            vx_size = torch.Tensor([1., 1., 1.])
        self.vx_size = vx_size if isinstance(vx_size, torch.Tensor) else torch.Tensor(vx_size)
        self.ref0 = ref0 if isinstance(ref0, torch.Tensor) else torch.Tensor(ref0)

        if ref_re is not None:
            self.ref_re = ref_re if isinstance(ref_re, torch.Tensor) else torch.Tensor(ref_re)
        else:
            self.ref_re = None

        self._cuda = cuda_kernel
        self.cuda_max_roi_chunk = cuda_max_roi_chunk

        self._init_spline_impl()
        self.sanity_check()

    def _init_spline_impl(self):
        """
        Init the spline implementation. Done seperately because otherwise it's harder to pickle

        """
        if self._cuda:
            self._spline_impl = spline_psf_cuda.PSFWrapperCUDA(self._coeff.shape[0], self._coeff.shape[1], self._coeff.shape[2],
                                                               self.roi_size_px[0], self.roi_size_px[1], self._coeff.numpy())
        else:
            self._spline_impl = spline_psf_cuda.PSFWrapperCPU(self._coeff.shape[0], self._coeff.shape[1], self._coeff.shape[2],
                                                              self.roi_size_px[0], self.roi_size_px[1], self._coeff.numpy())

    def sanity_check(self):
        """
        Perform some class specific safety checks
        Returns:

        """
        """Test whether extent corresponds to img shape"""
        if (self.img_shape[0] != (self.xextent[1] - self.xextent[0])) or \
                (self.img_shape[1] != (self.yextent[1] - self.yextent[0])):
            raise ValueError("Unequal size of extent and image shape not supported.")

    @property
    def cuda_is_available(self):
        return self._spline_impl.cuda_is_available

    @property
    def _ref_diff(self):
        if self.ref_re is None:
            return torch.zeros((1, 3))
        else:
            return self.ref_re - self.ref0

    @staticmethod
    def _cuda_compiled():
        """
        This is a dummy method to check whether CUDA is available without the need to init the class.
        Returns:

        """
        return spline_psf_cuda.PSFWrapperCPU(1, 1, 1, 1, 1, torch.zeros((1, 64)).numpy()).cuda_is_available

    @property
    def _roi_size_nm(self):
        roi_size_nm = (self.roi_size_px[0] * self.vx_size[0],
                       self.roi_size_px[1] * self.vx_size[1])

        return roi_size_nm

    @property
    def _cuda_max_drv_roi_chunk(self) -> int:
        # over 5 because 5 derivatives, over 2 because you return drv and roi
        return self.cuda_max_roi_chunk // (5 * 2)

    # # define pickles
    def __getstate__(self):
        """
        Returns dict without spline implementation attribute because C++ / CUDA implementation is not yet implemented
        to be pickleable itself. However, since the implementation is only accessed by this wrapper, this is not needed.

        """

        self_no_impl = dict(self.__dict__)
        del self_no_impl['_spline_impl']
        return self_no_impl

    def __setstate__(self, state):
        """
        Write dict and call init spline
        Args:
            state:

        Returns:

        """
        self.__dict__ = state
        self._init_spline_impl()

    def cuda(self):
        """
        Returns a copy of this object with implementation in CUDA. If already on CUDA, return original object.

        Returns:
            CubicSplinePSF instance

        """
        if self._cuda:
            return self

        return CubicSplinePSF(xextent=self.xextent, yextent=self.yextent, img_shape=self.img_shape, ref0=self.ref0,
                              coeff=self._coeff, vx_size=self.vx_size, roi_size=self.roi_size_px, cuda_kernel=True)

    def cpu(self):
        """
        Returns a copy of this object with implementation in CPU code. If already on CPU, return original object.

        Returns:
            CubicSplinePSF instance

        """
        if not self._cuda:
            return self

        return CubicSplinePSF(xextent=self.xextent, yextent=self.yextent, img_shape=self.img_shape, ref0=self.ref0,
                              coeff=self._coeff, vx_size=self.vx_size, roi_size=self.roi_size_px, cuda_kernel=False)

    def coord2impl(self, xyz):
        """
        Transforms nanometre coordinates to implementation coordiantes

        Args:
            xyz: (torch.Tensor)

        Returns:

        """
        offset = torch.Tensor([self.xextent[0] + 0.5, self.yextent[0] + 0.5, 0.]).float()
        return -xyz / self.vx_size + offset + self.ref0 - self._ref_diff

    def frame2roi_coord(self, xyz_nm: torch.Tensor):
        """
        Computes ROI wise coordinate from the coordinate on the frame and returns the px on the frame in addition

        Args:
            xyz_nm:

        Returns:
            xyz_r: roi-wise relative coordinates
            onframe_ix: ix where to place the roi on the final frame (caution: might be negative when roi is outside
            frame, but then we don't mean negative in pythonic sense)

        """
        xyz_r = xyz_nm.clone()
        """Get subpixel shift"""
        xyz_r[:, 0] = (xyz_r[:, 0] / self.vx_size[0]) % 1
        xyz_r[:, 1] = (xyz_r[:, 1] / self.vx_size[1]) % 1

        """Place emitters according to the Reference (reference is in px)"""
        xyz_r[:, :2] = (xyz_r[:, :2] + self.ref0[:2]) * self.vx_size[:2]
        xyz_px = (xyz_nm[:, :2] / self.vx_size[:2] - self.ref0[:2]).floor().int()

        return xyz_r, xyz_px

    def _forward_rois_impl(self, xyz, phot):
        """
        Computes the PSF and outputs the result ROI-wise.

        Args:
            xyz:
            phot:

        Returns:

        """

        n_rois = xyz.size(0)  # number of rois / emitters / fluorophores

        out = self._spline_impl.forward_rois(xyz[:, 0], xyz[:, 1], xyz[:, 2], phot)

        out = torch.from_numpy(out)
        out = out.reshape(n_rois, *self.roi_size_px)
        return out

    def forward_rois(self, xyz, phot):
        """
        Computes a ROI per position. The emitter is always centred as by the reference of the PSF; i.e. when working
        in px units, adding 1 in x or y direction does not change anything.

        Args:
            xyz: coordinates relative to the ROI centre.
            phot: photon count

        Returns:
            torch.Tensor with size N x roi_x x roi_y where N is the number of emitters / coordinates
                and roi_x/y the respective ROI size

        """
        if self.roi_size_px > self._roi_native:
            warnings.warn("You are trying to compute a ROI that is bigger than the "
                          "size supported by the spline coefficients.")

        xyz_, _ = self.frame2roi_coord(xyz)
        xyz_ = self.coord2impl(xyz_)

        return self._forward_rois_impl(xyz_, phot)

    def _forward_drv_chunks(self, xyz: torch.Tensor, weight: torch.Tensor, bg: torch.Tensor, add_bg: bool,
                            chunk_size: int):
        """Forwards the ROIs in chunks through CUDA in order not to let the GPU explode."""

        i = 0
        drv, rois = [], []
        while i <= len(xyz):
            slicer = slice(i, min(len(xyz), i + chunk_size))

            drv_, roi_ = self.derivative(xyz[slicer], weight[slicer], bg[slicer], add_bg=add_bg)
            drv.append(drv_)
            rois.append(roi_)
            i += chunk_size

        drv = torch.cat(drv, 0)
        rois = torch.cat(rois, 0)

        return drv, rois

    def derivative(self, xyz: torch.Tensor, phot: torch.Tensor, bg: torch.Tensor, add_bg: bool = True):
        """
        Computes the px wise derivative per ROI. Outputs ROIs additionally (since its computationally free of charge).
        The coordinates are (as the forward_rois method) relative to the reference of the PSF; i.e. when working
        in px units, adding 1 in x or y direction does not change anything.

        Args:
            xyz:
            phot:
            bg:
            add_bg (bool): add background value in ROI calculation

        Returns:
            derivatives (torch.Tensor): derivatives in correct units. Dimension N x N_par x H x W
            rois (torch.Tensor): ROIs. Dimension N x H x W
        """
        if xyz.size(0) == 0:  # fallback if there is no input
            return torch.zeros((0, 5, *self.roi_size_px)), torch.zeros((0, *self.roi_size_px))

        if self._cuda and self._cuda_max_drv_roi_chunk is not None and len(xyz) > self._cuda_max_drv_roi_chunk:
            return self._forward_drv_chunks(xyz, phot, bg, add_bg=add_bg, chunk_size=self._cuda_max_drv_roi_chunk)

        xyz_, _ = self.frame2roi_coord(xyz)
        xyz_ = self.coord2impl(xyz_)
        n_rois = xyz.size(0)

        drv_rois, rois = self._spline_impl.forward_drv_rois(xyz_[:, 0], xyz_[:, 1], xyz_[:, 2], phot, bg, add_bg)
        drv_rois = torch.from_numpy(drv_rois).reshape(n_rois, self.n_par, *self.roi_size_px)
        rois = torch.from_numpy(rois).reshape(n_rois, *self.roi_size_px)

        """Convert Implementation order to natural order, i.e. x/y/z/phot/bg instead of x/y/phot/bg/z."""
        drv_rois = drv_rois[:, [0, 1, 4, 2, 3]]

        """Apply correct units."""
        drv_rois[:, :3] /= self.vx_size.unsqueeze(-1).unsqueeze(-1)

        return drv_rois, rois

    def fisher(self, xyz: torch.Tensor, phot: torch.Tensor, bg: torch.Tensor):
        """
        Calculates the fisher matrix ROI wise. Outputs ROIs additionally (since its computationally free of charge).

        Args:
            xyz:
            phot:
            bg:

        Returns:
            fisher (torch.Tensor): Fisher Matrix. Dimension N x N_par x N_par
            rois (torch.Tensor): ROIs with background added. Dimension N x H x W
        """
        drv, rois = self.derivative(xyz, phot, bg, True)

        """
        Construct fisher by batched matrix multiplication. For this we have to play around with the axis of the
        derivatives.
        """
        drv_ = drv.permute(0, 2, 3, 1)
        fisher = torch.matmul(drv_.unsqueeze(-1), drv_.unsqueeze(-2))  # derivative contribution
        fisher = fisher / rois.unsqueeze(-1).unsqueeze(-1)  # px value contribution

        """Aggregate the drv along the pixel dimension"""
        fisher = fisher.sum(1).sum(1)

        return fisher, rois

    def crlb(self, xyz: torch.Tensor, phot: torch.Tensor, bg: torch.Tensor, inversion=None):
        """
        Computes the Cramer-Rao bound. Outputs ROIs additionally (since its computationally free of charge).

        Args:
            xyz:
            phot:
            bg:
            inversion: (function) overwrite default inversion with another function that can batch(!) invert matrices.
                The last two dimensions are the the to be inverted dimensions. Dimension of fisher matrix: N x H x W
                where N is the batch dimension.

        Returns:
            crlb (torch.Tensor): Cramer-Rao-Lower Bound. Dimension N x N_par
            rois (torch.Tensor): ROIs with background added. Dimension N x H x W
        """

        if inversion is not None:
            inv_f = inversion
        else:
            inv_f = self.inv_default

        fisher, rois = self.fisher(xyz, phot, bg)
        fisher_inv = inv_f(fisher)
        crlb = torch.diagonal(fisher_inv, dim1=1, dim2=2)

        return crlb, rois

    def crlb_sq(self, xyz: torch.Tensor, phot: torch.Tensor, bg: torch.Tensor, inversion=None):
        """
        Function for the lazy ones to compute the sqrt Cramer-Rao bound. Outputs ROIs additionally (since its
        computationally free of charge).

        Args:
            xyz:
            phot:
            bg:
            inversion: (function) overwrite default inversion with another function that can batch invert matrices

        Returns:
            crlb (torch.Tensor): Cramer-Rao-Lower Bound. Dimension N x N_par
            rois (torch.Tensor): ROIs with background added. Dimension N x H x W
        """
        crlb, rois = self.crlb(xyz, phot, bg, inversion)
        return crlb.sqrt(), rois

    def _forward_chunks(self, xyz: torch.Tensor, weight: torch.Tensor, frame_ix: torch.Tensor,
                        ix_low: int, ix_high: int, chunk_size: int):

        i = 0
        f = torch.zeros((ix_high - ix_low + 1, *self.img_shape))
        while i <= len(xyz):
            slicer = slice(i, min(len(xyz), i + chunk_size))

            f += self.forward(xyz[slicer], weight[slicer], frame_ix[slicer], ix_low, ix_high)
            i += chunk_size

        return f

    def forward(self, xyz: torch.Tensor, weight: torch.Tensor, frame_ix: torch.Tensor = None, ix_low: int = None,
                ix_high: int = None):
        """
        Forward coordinates frame index aware through the psf model.

        Args:
            xyz: coordinates of size N x (2 or 3)
            weight: photon value
            frame_ix: (optional) frame index
            ix_low: (optional) lower frame_index, if None will be determined automatically
            ix_high: (optional) upper frame_index, if None will be determined automatically

        Returns:
            frames: (torch.Tensor)
        """
        xyz, weight, frame_ix, ix_low, ix_high = super().forward(xyz, weight, frame_ix, ix_low, ix_high)

        if xyz.size(0) == 0:
            return torch.zeros((ix_high - ix_low + 1, *self.img_shape))

        if self._cuda and self.cuda_max_roi_chunk is not None and len(xyz) > self.cuda_max_roi_chunk:
            return self._forward_chunks(xyz, weight, frame_ix, ix_low, ix_high, self.cuda_max_roi_chunk)

        """Convert Coordinates into ROI based coordinates and transform into implementation coordinates"""
        xyz_r, ix = self.frame2roi_coord(xyz)
        xyz_r = self.coord2impl(xyz_r)

        n_frames = ix_high - ix_low + 1

        frames = self._spline_impl.forward_frames(*self.img_shape,
                                                  frame_ix,
                                                  n_frames,
                                                  xyz_r[:, 0],
                                                  xyz_r[:, 1],
                                                  xyz_r[:, 2],
                                                  ix[:, 0],
                                                  ix[:, 1],
                                                  weight)

        frames = torch.from_numpy(frames).reshape(n_frames, *self.img_shape)
        return frames
