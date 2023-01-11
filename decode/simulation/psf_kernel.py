import math
import warnings
from abc import ABC
from typing import Tuple, Optional

import numpy as np
import spline  # cubic spline implementation
import torch
try:
    import zernike
except ImportError:
    zernike = None

import decode.generic.utils
from ..generic import slicing as gutil
from ..generic import deploy


class PSF(ABC):
    def __init__(
            self,
            xextent: Tuple[float, float] = (None, None),
            yextent: Tuple[float, float] = (None, None),
            zextent: Tuple[float, float] = (None, None),
            img_shape: Tuple[int, int] = (None, None),
    ):
        """
        Abstract class to represent a point spread function.
        `.forward` must be overwritten and shall be called via super().forward(...)
        before the subclass implementation follows.

        Args:
            xextent: extent of the psf in x
            yextent: extent of the psf in y
            zextent: extent of the psf in z
            img_shape: shape of the output frame
        """

        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.img_shape = img_shape

    def __str__(self):
        return "PSF: \n xextent: {}\n yextent: {}\n zextent: {}\n img_shape: {}".format(
            self.xextent, self.yextent, self.zextent, self.img_shape
        )

    def crlb(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
            self,
            xyz: torch.Tensor,
            weight: torch.Tensor,
            frame_ix: Optional[torch.Tensor] = None,
            ix_low: Optional[int] = None,
            ix_high: Optional[int] = None,
    ):
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

        # NOTE: This is the most basic implementation (probably inefficient for your PSF).
        # The frame indices are shifted, the emtiters are filtered and then the frames
        # are computed one by one. Better implement a batched version of that.
        xyz, weight, frame_ix, ix_low, ix_high = self._auto_filter_shift(
            xyz=xyz, weight=weight, frame_ix=frame_ix, ix_low=ix_low, ix_high=ix_high
        )
        return self._forward_single_frame_wrapper(
            xyz, weight, frame_ix, ix_low, ix_high
        )

    @staticmethod
    def _auto_filter_shift(
            xyz: torch.Tensor,
            weight: torch.Tensor,
            frame_ix: torch.Tensor,
            ix_low: int,
            ix_high: int,
    ):
        # filter emitters by frame and shift frame index to start at 0

        if frame_ix is None:
            frame_ix = torch.zeros((xyz.size(0),)).int()

        if ix_low is None:
            ix_low = frame_ix.min().item()

        if ix_high is None:
            ix_high = frame_ix.max().item() + 1  # pythonic

        # kick out everything that is out of frame index bounds and shift the frames to start at 0
        in_frame = (frame_ix >= ix_low) * (frame_ix < ix_high)
        xyz_ = xyz[in_frame, :]
        weight_ = weight[in_frame] if weight is not None else None
        frame_ix_ = frame_ix[in_frame] - ix_low

        ix_high = ix_high - ix_low
        ix_low = 0

        return xyz_, weight_, frame_ix_, ix_low, ix_high

    def _forward_single_frame(self, xyz: torch.Tensor, weight: torch.Tensor):
        raise NotImplementedError

    def _forward_single_frame_wrapper(
            self,
            xyz: torch.Tensor,
            weight: torch.Tensor,
            frame_ix: torch.Tensor,
            ix_low: int,
            ix_high: int,
    ):
        """
        Convenience (fallback) wrapper that splits the input in frames and computes the
        frames one by one. Use only if the forward implemenation is not batched.

        Args:
            xyz: coordinates of size N x (2 or 3)
            weight: photon value
            frame_ix: frame index
            ix_low (int):  lower frame_index, if None will be determined automatically
            ix_high (int):  upper frame_index, if None will be determined automatically

        Returns:
            frames (torch.Tensor): N x H x W, stacked frames
        """
        ix_split = gutil.ix_split(frame_ix, ix_min=ix_low, ix_max=ix_high)
        n_splits = len(ix_split)

        if weight is not None:
            frames = [
                self._forward_single_frame(xyz[ix_split[i]], weight[ix_split[i]])
                for i in range(n_splits)
            ]
        else:
            frames = [
                self._forward_single_frame(xyz[ix_split[i]], None)
                for i in range(n_splits)
            ]

        if len(frames) == 0:
            frames = torch.zeros(0, *self.img_shape)
        else:
            frames = torch.stack(frames, 0)
        return frames


class DeltaPSF(PSF):
    def __init__(
            self,
            xextent: Tuple[float, float],
            yextent: Tuple[float, float],
            img_shape: Tuple[int, int],
    ):
        """
        Delta function PSF. You input a list of coordinates, this class outputs a single
        one-hot representation in 2D of your input.
        If multiple things fall into the same bin, the output is the weight of either
        of the two (which one is arbitrary implementation detail).

        """
        super().__init__(xextent=xextent, yextent=yextent, img_shape=img_shape)

        from decode.emitter.process import EmitterFilterFoV

        self._fov_filter = EmitterFilterFoV(
            xextent=self.xextent, yextent=self.yextent, zextent=None
        )

        (
            self._bin_x,
            self._bin_y,
            self._bin_ctr_x,
            self._bin_ctr_y,
        ) = decode.generic.utils.frame_grid(img_shape, xextent, yextent)

    @property
    def bin_ctr_x(self):
        """
        Read only bin_ctr_x
        """
        return self._bin_ctr_x

    @property
    def bin_ctr_y(self):
        """
        Read only bin_ctr_y
        """
        return self._bin_ctr_y

    def search_bin_index(self, xy: torch.Tensor, raise_outside: bool = True):
        """
        Returns the index of the bin in question, x ix and y ix.
        Make sure items are actually fit in the bins (i.e. filter outside ones before) or handle those items later on.

        Args:
            xy: xy coordinates
            raise_outside: raise error if anything is outside of the specified bins; otherwise those coordinate's
                indices are -1 and len(bin) which means outside of the bin's range.

        """

        x_ix = np.searchsorted(self._bin_x, xy[:, 0], side="right") - 1
        y_ix = np.searchsorted(self._bin_y, xy[:, 1], side="right") - 1

        if raise_outside:
            if (
                    ~(
                            (x_ix >= 0)
                            * (x_ix <= len(self._bin_x) - 2)
                            * (y_ix >= 0)
                            * (y_ix <= len(self._bin_y) - 2)
                    )
            ).any():
                raise ValueError(
                    "At least one value outside of the specified bin ranges."
                )

        return x_ix, y_ix

    def forward(
            self,
            xyz: torch.Tensor,
            weight: torch.Tensor,
            frame_ix: Optional[torch.Tensor] = None,
            ix_low: Optional[int] = None,
            ix_high: Optional[int] = None,
    ):
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
        if weight is None:
            weight = torch.ones_like(xyz[:, 0])

        xyz, weight, frame_ix, ix_low, ix_high = self._auto_filter_shift(
            xyz, weight, frame_ix, ix_low, ix_high
        )

        # remove Emitters that are out of the frame
        mask = self._fov_filter._clean_emitter(xyz)

        x_ix, y_ix = self.search_bin_index(xyz[mask], raise_outside=True)
        n_ix = frame_ix[mask].long()

        # compute frames
        frames = torch.zeros((ix_high - ix_low, *self.img_shape))
        frames[n_ix, x_ix, y_ix] = weight[mask]

        return frames


class GaussianPSF(PSF):
    def __init__(
            self,
            xextent: Tuple[float, float],
            yextent: Tuple[float, float],
            zextent: Tuple[float, float],
            img_shape: Tuple[int, int],
            sigma_0: float,
            peak_weight: bool = False,
    ):
        """
        Gaussian PSF Model.If no z extent is provided we assume a 2D PSF.

        Args:
            xextent: (tuple of float) extent of psf in x
            yextent: (tuple of float) extent of psf in y
            zextent: (tuple of float or None, optional) extent of psf in z
            img_shape: (tuple) img shape
            sigma_0: sigma in focus in px
            peak_weight: (bool) if true: use peak intensity instead of integral under the curve
        """
        super().__init__(
            xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape
        )

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

        sigma_x = sigma_0 * (1 + ((z + foc_shift) / (rl_range)) ** 2).sqrt()
        sigma_y = sigma_0 * (1 + ((z - foc_shift) / (rl_range)) ** 2).sqrt()

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

        gauss_x = torch.erf(
            (xx[1:, 1:, :] - xpos) / (math.sqrt(2) * sig_x)
        ) - torch.erf((xx[0:-1, 1:, :] - xpos) / (math.sqrt(2) * sig_x))

        gauss_y = torch.erf(
            (yy[1:, 1:, :] - ypos) / (math.sqrt(2) * sig_y)
        ) - torch.erf((yy[1:, 0:-1, :] - ypos) / (math.sqrt(2) * sig_y))

        gaussCdf = weight.type_as(gauss_x) / 4 * torch.mul(gauss_x, gauss_y)
        if self.peak_weight:
            gaussCdf *= 2 * math.pi * sig_x * sig_y
        gaussCdf = torch.sum(gaussCdf, 2)

        return gaussCdf


class CubicSplinePSF(PSF):
    n_par = 5  # x, y, z, phot, bg
    inv_default = torch.inverse

    def __init__(
            self,
            xextent: Tuple[float, float],
            yextent: Tuple[float, float],
            img_shape: Tuple[int, int],
            ref0: tuple,
            coeff: torch.Tensor,
            vx_size: Tuple[int, int, int],
            *,
            roi_size: (None, tuple) = None,
            ref_re: (None, torch.Tensor, tuple) = None,
            roi_auto_center: bool = False,
            device: str = "cpu",
            max_roi_chunk: int = 500000,
    ):
        """
        Cubic spline PSF, i.e. a step-wise polynomial with coefficients of size X,Y,Z,64.

        Args:
            ref_re:
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape (tuple): img_shape
            coeff: spline coefficients
            ref0 (tuple): zero reference point in implementation units
            vx_size (tuple): pixel / voxel size
            roi_size (tuple, None, optional): roi_size. optional. can be determined from
                dimension of coefficients.
            device: specify the device for the implementation to run on. Must be like
                ('cpu', 'cuda', 'cuda:1')
            max_roi_chunk (int): max number of rois to be processed at a time via the
                cuda kernel. If you run into memory allocation errors, decrease this
                number or free some space on your CUDA device.
        """
        super().__init__(
            xextent=xextent, yextent=yextent, zextent=None, img_shape=img_shape
        )

        self._coeff = coeff
        self._roi_native = self._coeff.size()[
                           :2
                           ]  # native roi based on the coeff's size
        self.roi_size_px = (
            torch.Size(roi_size) if roi_size is not None else self._roi_native
        )

        if vx_size is None:
            vx_size = torch.Tensor([1.0, 1.0, 1.0])

        self.vx_size = (
            vx_size if isinstance(vx_size, torch.Tensor) else torch.Tensor(vx_size)
        )
        self.ref0 = ref0 if isinstance(ref0, torch.Tensor) else torch.Tensor(ref0)

        self.ref_re = self._shift_ref(ref_re, roi_auto_center)

        self._device, self._device_ix = decode.utils.hardware._specific_device_by_str(
            device
        )
        self.max_roi_chunk = max_roi_chunk

        self._init_spline_impl()
        self.sanity_check()

    def _shift_ref(self, ref_re, auto_center):

        if ref_re is not None and auto_center:
            raise ValueError(
                "PSF reference can not be automatically centered when you specify a"
                " custom centre at the same time."
            )

        elif auto_center:
            if self.roi_size_px[0] % 2 != 1 or self.roi_size_px[1] % 2 != 1:
                raise ValueError(
                    "PSF reference can not be centered when the roi_size is even."
                )

            return torch.Tensor(
                [
                    (self.roi_size_px[0] - 1) // 2,
                    (self.roi_size_px[1] - 1) // 2,
                    self.ref0[2],
                ]
            )

        elif ref_re is not None:
            return ref_re if isinstance(ref_re, torch.Tensor) else torch.Tensor(ref_re)

    def _init_spline_impl(self):
        """
        Init the spline implementation. Done seperately because otherwise it's harder to pickle
        """
        if "cuda" in self._device:
            if self._device_ix is None:
                device_ix = 0
            else:
                device_ix = self._device_ix

            self._spline_impl = spline.PSFWrapperCUDA(
                self._coeff.shape[0],
                self._coeff.shape[1],
                self._coeff.shape[2],
                self.roi_size_px[0],
                self.roi_size_px[1],
                self._coeff.numpy(),
                device_ix,
            )
        elif "cpu" == self._device:
            self._spline_impl = spline.PSFWrapperCPU(
                self._coeff.shape[0],
                self._coeff.shape[1],
                self._coeff.shape[2],
                self.roi_size_px[0],
                self.roi_size_px[1],
                self._coeff.numpy(),
            )
        else:
            raise ValueError(f"Unsupported device ({self._device} has been set.")

    def sanity_check(self):
        """
        Perform some class specific safety checks
        Returns:

        """
        # test whether extent corresponds to img shape
        if (self.img_shape[0] != (self.xextent[1] - self.xextent[0])) or (
                self.img_shape[1] != (self.yextent[1] - self.yextent[0])
        ):
            raise ValueError("Unequal size of extent and image shape not supported.")

        if (torch.tensor(self.roi_size_px) > torch.tensor(self._roi_native)).any():
            warnings.warn(
                "The specified ROI size is larger than the size supported by "
                "the spline coefficients. While this mostly likely works "
                "computationally, results may be unexpected."
            )

    @property
    def cuda_compiled(self) -> bool:
        """
        Returns true if (1) a CUDA capable device is available and (2) spline was compiled with CUDA support.
        Technically (1) could come without (2).

        """
        return spline.cuda_compiled

    @staticmethod
    def cuda_is_available() -> bool:
        """
        This is a dummy method to check whether CUDA is available without the need to
        init the class. I wonder whether Python has 'static properties'?

        """
        return spline.cuda_is_available()

    @property
    def _ref_diff(self):
        if self.ref_re is None:
            return torch.zeros((3))
        else:
            return self.ref_re - self.ref0

    @property
    def _roi_size_nm(self):
        roi_size_nm = (
            self.roi_size_px[0] * self.vx_size[0],
            self.roi_size_px[1] * self.vx_size[1],
        )

        return roi_size_nm

    @property
    def _max_drv_roi_chunk(self) -> int:
        # over 5 because 5 derivatives, over 2 because you return drv and roi
        return self.max_roi_chunk // (5 * 2)

    # pickle customisation
    def __getstate__(self):
        """
        Returns dict without spline implementation attribute because C++ / CUDA implementation is not (yet) implemented
        to be pickleable itself. However, since the CUDA/C++ implementation is only accessed by this wrapper, this is
        not strictly needed. This class becomes pickleable by excluding the spline implementation and rather re-init
        the implementation every time it is unpickled.

        """

        self_no_impl = dict(self.__dict__)
        del self_no_impl["_spline_impl"]
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

    def cuda(self, ix: int = 0):
        """
        Returns a copy of this object with implementation in CUDA. If already on CUDA and selected device, return original object.

        Args:
            ix: device index

        Returns:
            CubicSplinePSF instance

        """
        if "cuda" in self._device and ix == self._device_ix:
            return self

        return CubicSplinePSF(
            xextent=self.xextent,
            yextent=self.yextent,
            img_shape=self.img_shape,
            ref0=self.ref0,
            coeff=self._coeff,
            vx_size=self.vx_size,
            roi_size=self.roi_size_px,
            device=f"cuda:{ix}",
        )

    def cpu(self):
        """
        Returns a copy of this object with implementation in CPU code. If already on CPU, return original object.

        Returns:
            CubicSplinePSF instance

        """
        if self._device == "cpu":
            return self

        return CubicSplinePSF(
            xextent=self.xextent,
            yextent=self.yextent,
            img_shape=self.img_shape,
            ref0=self.ref0,
            coeff=self._coeff,
            vx_size=self.vx_size,
            roi_size=self.roi_size_px,
            device="cpu",
        )

    def coord2impl(self, xyz):
        """
        Transforms nanometre coordinates to implementation coordiantes

        Args:
            xyz: (torch.Tensor)

        Returns:

        """
        offset = torch.Tensor(
            [self.xextent[0] + 0.5, self.yextent[0] + 0.5, 0.0]
        ).float()
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
        # get subpixel shift
        xyz_r[:, 0] = (xyz_r[:, 0] / self.vx_size[0]) % 1
        xyz_r[:, 1] = (xyz_r[:, 1] / self.vx_size[1]) % 1

        # place emitters according to the Reference (reference is in px)
        xyz_r[:, :2] = (xyz_r[:, :2] + self.ref0[:2]) * self.vx_size[:2]
        xyz_px = (
            (xyz_nm[:, :2] / self.vx_size[:2] - self.ref0[:2] - self._ref_diff[:2])
            .floor()
            .int()
        )

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
        Computes a ROI per position. The emitter is always centred as by the reference
        of the PSF; i.e. when working in px units, adding 1 in x or y direction does
        not change anything.

        Args:
            xyz: coordinates relative to the ROI centre.
            phot: photon count

        Returns:
            torch.Tensor with size N x roi_x x roi_y where N is the number of emitters / coordinates
                and roi_x/y the respective ROI size

        """

        xyz_, _ = self.frame2roi_coord(xyz)
        xyz_ = self.coord2impl(xyz_)

        return self._forward_rois_impl(xyz_, phot)

    def _forward_drv_chunks(
            self,
            xyz: torch.Tensor,
            weight: torch.Tensor,
            bg: torch.Tensor,
            add_bg: bool,
            chunk_size: int,
    ):
        """Forwards the ROIs in chunks through CUDA in order not to let the GPU explode."""

        i = 0
        drv, rois = [], []
        while i <= len(xyz):
            slicer = slice(i, min(len(xyz), i + chunk_size))

            drv_, roi_ = self.derivative(
                xyz[slicer], weight[slicer], bg[slicer], add_bg=add_bg
            )
            drv.append(drv_)
            rois.append(roi_)
            i += chunk_size

        drv = torch.cat(drv, 0)
        rois = torch.cat(rois, 0)

        return drv, rois

    def derivative(
            self,
            xyz: torch.Tensor,
            phot: torch.Tensor,
            bg: torch.Tensor,
            add_bg: bool = True,
    ):
        """
        Computes the px wise derivative per ROI. Outputs ROIs additionally
        (since its computationally free of charge). The coordinates are
        (as the forward_rois method) relative to the reference of the PSF; i.e.
        when working in px units, adding 1 in x or y direction does not change
        anything.

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
            return torch.zeros((0, 5, *self.roi_size_px)), torch.zeros(
                (0, *self.roi_size_px)
            )

        if self._max_drv_roi_chunk is not None and len(xyz) > self._max_drv_roi_chunk:
            return self._forward_drv_chunks(
                xyz, phot, bg, add_bg=add_bg, chunk_size=self._max_drv_roi_chunk
            )

        xyz_, _ = self.frame2roi_coord(xyz)
        xyz_ = self.coord2impl(xyz_)
        n_rois = xyz.size(0)

        drv_rois, rois = self._spline_impl.forward_drv_rois(
            xyz_[:, 0], xyz_[:, 1], xyz_[:, 2], phot, bg, add_bg
        )
        drv_rois = torch.from_numpy(drv_rois).reshape(
            n_rois, self.n_par, *self.roi_size_px
        )
        rois = torch.from_numpy(rois).reshape(n_rois, *self.roi_size_px)

        # convert Implementation order to natural order,
        # i.e. x/y/z/phot/bg instead of x/y/phot/bg/z."""
        drv_rois = drv_rois[:, [0, 1, 4, 2, 3]]

        # apply correct units
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

        # construct fisher by batched matrix multiplication. For this we have to play
        # around with the axis of the derivatives.
        drv_ = drv.permute(0, 2, 3, 1)
        fisher = torch.matmul(
            drv_.unsqueeze(-1), drv_.unsqueeze(-2)
        )  # derivative contribution
        fisher = fisher / rois.unsqueeze(-1).unsqueeze(-1)  # px value contribution

        # aggregate the drv along the pixel dimension
        fisher = fisher.sum(1).sum(1)

        return fisher, rois

    def crlb(
            self, xyz: torch.Tensor, phot: torch.Tensor, bg: torch.Tensor,
            inversion=None
    ):
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

    def crlb_sq(
            self, xyz: torch.Tensor, phot: torch.Tensor, bg: torch.Tensor,
            inversion=None
    ):
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

    def _forward_chunks(
            self,
            xyz: torch.Tensor,
            weight: torch.Tensor,
            frame_ix: torch.Tensor,
            ix_low: int,
            ix_high: int,
            chunk_size: int,
    ):

        i = 0
        f = torch.zeros((ix_high - ix_low, *self.img_shape))
        while i <= len(xyz):
            slicer = slice(i, min(len(xyz), i + chunk_size))

            f += self.forward(
                xyz[slicer], weight[slicer], frame_ix[slicer], ix_low, ix_high
            )
            i += chunk_size

        return f

    def forward(
            self,
            xyz: torch.Tensor,
            weight: torch.Tensor,
            frame_ix: Optional[torch.Tensor] = None,
            ix_low: Optional[int] = None,
            ix_high: Optional[int] = None,
    ):
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
        xyz, weight, frame_ix, ix_low, ix_high = self._auto_filter_shift(
            xyz, weight, frame_ix, ix_low, ix_high
        )

        if xyz.size(0) == 0:
            return torch.zeros((ix_high - ix_low, *self.img_shape))

        if self.max_roi_chunk is not None and len(xyz) > self.max_roi_chunk:
            return self._forward_chunks(
                xyz, weight, frame_ix, ix_low, ix_high, self.max_roi_chunk
            )

        # convert Coordinates into ROI based coordinates and transform
        # into implementation coordinates
        xyz_r, ix = self.frame2roi_coord(xyz)
        xyz_r = self.coord2impl(xyz_r)

        n_frames = ix_high - ix_low

        frames = self._spline_impl.forward_frames(
            *self.img_shape,
            frame_ix,
            n_frames,
            xyz_r[:, 0],
            xyz_r[:, 1],
            xyz_r[:, 2],
            ix[:, 0],
            ix[:, 1],
            weight,
        )

        frames = torch.from_numpy(frames).reshape(n_frames, *self.img_shape)
        return frames


class ZernikePSF(PSF):
    @deploy.raise_optional_deps("zernike", "please install zernike via `pip install zernike`")
    def __init__(
            self,
            xextent,
            yextent,
            zextent,
            img_shape: tuple[int, int],
            *,
            weight_real: list,
            weight_phase: list,
            # theta: float,  # can be absorbed by zernike polynomial itself
            num_aperture: float,
            ref_index: float,
            wavelength: float,
            k_size: int,
            zernike_order: int,
            voxel_size: tuple[float, float, float],
    ):
        """
        Zernike vectorial based PSF model.


        Args:
            xextent:
            yextent:
            zextent:
            weight_real: weights of the real part of the zernike polynomials
            weight_phase: weights of the imaginary part of the zernike polynomials
            theta:
            num_aperture:
            ref_index:
            wavelength:
            k_size:
            voxel_size: in units of wavelength e.g. (110, 120, 50)
                if the intput is entirely in px units
        """

        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent)

        self._img_shape = img_shape
        self._pramp_x = None
        self._pramp_y = None
        self._pupil_e_field = None

        self._weight_real = torch.tensor(weight_real)
        self._weight_phase = torch.tensor(weight_phase)
        self._theta = 1.
        self._t_a = 1.
        self._num_aperture = num_aperture
        self._ref_index = ref_index
        self._wavelength = wavelength
        self._k_size = k_size
        self._zernike_order = zernike_order
        self._voxel_size = torch.tensor(voxel_size)

        self._pramp_x, self._pramp_y = self._get_phase_ramp()
        self._zernike_pol = self._get_zernike_poly()
        self._zernike_real = self._get_zernike_real()
        self._zernike_phase = self._get_zernike_phase()
        self._kx, self._ky, self._kz = self._get_kxyz()
        self._k = ref_index / wavelength

        unit_x, unit_y = self._get_unit_vectors()
        # decompose wave vector into radial and z component
        self._kr = (unit_x**2 + unit_y**2).sqrt() * num_aperture / wavelength
        self._pupil_radius = 1.

        # for now scalar model
        self._emn = 1.

    def _get_zernike_poly(self) -> torch.Tensor:
        """
        Compute zernike polynomials of size k_size x k_size up to zernike_order
        """
        p = zernike.RZern(self._zernike_order)
        ddx = np.linspace(-1.0, 1.0, self._k_size)
        ddy = np.linspace(-1.0, 1.0, self._k_size)
        xv, yv = np.meshgrid(ddx, ddy)

        p.make_cart_grid(xv, yv)
        n_pol = p.nk

        zpol = []
        for i in range(n_pol):
            c = np.zeros(n_pol)
            c[i] = 1.
            phi = p.eval_grid(c, matrix=True)
            zpol.append(torch.from_numpy(phi))

        return torch.stack(zpol, 0)

    def _get_phase_ramp(self):
        ramp_x, ramp_y = torch.meshgrid(
            torch.linspace(0, 2 * math.pi, self._img_shape[0]),
            torch.linspace(0, 2 * math.pi, self._img_shape[1]),
        )

        return ramp_x, ramp_y

    def _get_zernike_real(self) -> torch.Tensor:
        """Get zernike polynomial weighted by respective amplitude (real)"""
        return (self._weight_real * self._zernike_pol).sum(0)

    def _get_zernike_phase(self) -> torch.Tensor:
        """Get zernike polynomial weighted by respective amplitude (phase)"""
        return (self._weight_phase * self._zernike_pol).sum(0)

    def _get_kxyz(self):
        """Get k vectors in x, y and z (fourier space, inverse of micron units)"""
        krange = np.linspace(-self._pupil_radius + self._pupil_radius / self._k_size,
                             self._pupil_radius - self._pupil_radius / self._k_size, self._k_size)

        [yy, xx] = np.meshgrid(krange, krange)
        kr = np.lib.scimath.sqrt((xx ** 2 + yy ** 2))
        # pupilmag = np.complex64(kr < 1)
        # cos_nimm = np.lib.scimath.sqrt(1 - (kr * self._num_aperture / self._ref_index) ** 2)

        kx = np.complex64(xx * self._num_aperture / self._wavelength)
        ky = np.complex64(yy * self._num_aperture / self._wavelength)
        kz = np.lib.scimath.sqrt(
            (self._ref_index / self._wavelength) ** 2 - (kr * self._num_aperture / self._wavelength) ** 2)

        return kx, ky, kz

    def _get_unit_vectors(self):
        kx, ky = torch.linspace(-1, 1, self._k_size), torch.linspace(-1, 1, self._k_size)
        kx, ky = torch.meshgrid(kx, ky)
        return kx, ky

    @staticmethod
    def pupile_field(
            na: float,
            na_sample: float,
            na_oil: float,
            na_cover: float,
            k_size: int,
            pupilradius: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute eletric field in pupile space

        Args:
            na: refractive index
            na_sample: refractive index of sample medium
            na_oil: refractive index of oil
            na_cover: refractive index of cover slip
            k_size: size of frequency domain (i.e. in pupil space)
            pupilradius:

        Returns:
            tensor of size k_size x k_size

        """
        krange = np.linspace(
            -pupilradius + pupilradius / k_size,
            pupilradius - pupilradius / k_size,
            k_size,
        )
        [xx, yy] = np.meshgrid(krange, krange)
        kr = np.lib.scimath.sqrt(xx ** 2 + yy ** 2)

        cos_imm = np.lib.scimath.sqrt(1 - (kr * na / na_oil) ** 2)
        cos_med = np.lib.scimath.sqrt(1 - (kr * na / na_sample) ** 2)
        cos_cov = np.lib.scimath.sqrt(1 - (kr * na / na_cover) ** 2)

        FresnelPmedcov = (
                2 * na_sample * cos_med / (na_sample * cos_cov + na_cover * cos_med)
        )
        FresnelSmedcov = (
                2 * na_sample * cos_med / (na_sample * cos_med + na_sample * cos_med)
        )
        FresnelPcovimm = (
                2 * na_cover * cos_cov / (na_cover * cos_imm + na_oil * cos_cov)
        )
        FresnelScovimm = (
                2 * na_cover * cos_cov / (na_cover * cos_cov + na_oil * cos_imm)
        )
        Tp = FresnelPmedcov * FresnelPcovimm
        Ts = FresnelSmedcov * FresnelScovimm

        phi = np.arctan2(yy, xx)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        sin_med = kr * na / na_sample

        pvec = Tp * np.stack([cos_med * cos_phi, cos_med * sin_phi, -sin_med])
        svec = Ts * np.stack([-sin_phi, cos_phi, np.zeros(cos_phi.shape)])

        hx = cos_phi * pvec - sin_phi * svec
        hy = sin_phi * pvec + cos_phi * svec
        h = np.concatenate((hx, hy), axis=0)  # electric field

        return torch.from_numpy(h)

    def _forward_single_frame(self, xyz: torch.Tensor, weight: torch.Tensor):
        obj_fn = 1.
        smoothing = 1.
        bg = 0.

        xyz *= self._voxelsize
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # get from _get_zernike_real
        h = self._t_a * self._zernike_real * torch.exp(1j * self._zernike_phase)
        h_k = h * self._e_mn * torch.exp(2j * math.pi * (self._kx * x + self._ky * y + self._kz * z))
        u = self.fft(h_k).abs() ** 2  # missing sum
        real = u * weight * obj_fn * smoothing + bg

        return real


class VoxelatedPSF(PSF):
    # ToDo: cube based psf. Needs 3D voxel cube as input
    #  and sufficient interpolation (e.g. fft).
    pass
