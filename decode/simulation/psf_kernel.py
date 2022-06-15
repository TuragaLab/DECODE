import math
import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

import numpy as np
import spline  # cubic spline implementation
import torch

from decode.generic import slicing as gutil
import decode.generic.utils

from torch.jit import script
import numba
from numba import cuda

@cuda.jit
def place_roi(frames, roi_grads, frame_s_b, frame_s_c, frame_s_z, frame_s_y, frame_s_x, rois, roi_s_n, roi_s_z, roi_s_y, roi_s_x, b, c, z, y, x):
    
    kx = cuda.grid(1)
    # One thread for every pixel in the roi stack. Exit if outside
    if kx >= roi_s_n * roi_s_z * roi_s_y * roi_s_x: 
        return
    
    # roi index
    xir = kx % roi_s_x; kx = kx // roi_s_x
    yir = kx % roi_s_y; kx = kx // roi_s_y
    zir = kx % roi_s_z; kx = kx // roi_s_z
    nir = kx % roi_s_n 
    
    # frame index
    bif = b[nir]
    cif = c[nir]
    zif = z[nir] + zir
    yif = y[nir] + yir
    xif = x[nir] + xir
    
    if ((bif < 0) or (bif >= frame_s_b)): return
    if ((cif < 0) or (cif >= frame_s_c)): return
    if ((zif < 0) or (zif >= frame_s_z)): return
    if ((yif < 0) or (yif >= frame_s_y)): return
    if ((xif < 0) or (xif >= frame_s_x)): return
    
    cuda.atomic.add(frames, (bif, cif, zif, yif, xif), rois[nir, zir, yir, xir])
    # The gradients for the ROIs are just one if they are inside the frames and 0 otherwise. Easy to do here and then just ship to the backward function
    roi_grads[nir, zir, yir, xir] = 1
    # Alternative to atomic.add. No difference in speed
#     frames[bif, cif, zif, yif, xif] += rois[nir, zir, yir, xir]

"""THIS FUNCTION BREAKS AUTORELOAD FOR SOME REASON? https://discuss.pytorch.org/t/class-autograd-function-in-module-cause-autoreload-fail-in-jupyter-lab/96250/2"""
class CudaPlaceROI(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, rois, frame_s_b, frame_s_c, frame_s_z, frame_s_y, frame_s_x, roi_s_n, roi_s_z, roi_s_y, roi_s_x, b, c, z, y, x):

        frames = torch.zeros([frame_s_b, frame_s_c, frame_s_z, frame_s_y, frame_s_x]).to('cuda')
        rois_grads = torch.zeros([roi_s_n, roi_s_z, roi_s_y, roi_s_x]).to('cuda')
        
        threadsperblock = 256
        blocks = ((roi_s_n * roi_s_z * roi_s_y * roi_s_x) + (threadsperblock - 1)) // threadsperblock        
        place_roi[blocks, threadsperblock](frames, rois_grads, frame_s_b, frame_s_c, frame_s_z, frame_s_y, frame_s_x, rois.detach(), roi_s_n, roi_s_z, roi_s_y, roi_s_x, b, c, z, y, x)
        
        ctx.save_for_backward(rois_grads)
        
        return frames
    
    @staticmethod
    def backward(ctx, grad_output):
        rois_grads, = ctx.saved_tensors
        return rois_grads, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
@script
def _place_psf(psf_vols, b, ch, z, y, x, output_shape):
    
    """jit function for placing PSFs
    1) This function will add padding to coordinates (z, y, x) (we need padding in order to place psf on the edges)
    afterwards we will just crop out to original shape
    2) Create empty tensor with paddings loc3d_like
    3) place each individual PSFs in to the corresponding cordinates in loc3d_like
    4) unpad to original output shape

    Args:
        psf_vols:   torch.Tensor
        b:        torch.Tensor
        c:        torch.Tensor
        h:        torch.Tensor
        w:        torch.Tensor
        d:        torch.Tensor
        szs:      torch.Tensor
        
    Shape:
        psf_vols: (Num_E, C, PSF_SZ_X, PSF_SZ_Y, PSF_SZ_Z)
        b:  (Num_E,)
        c:  (Num_E,)
        h:  (Num_E,)
        w:  (Num_E,)
        d:  (Num_E,)
        output_shape:  (BS, Frames, H, W, D)
        
    -Output: placed_psf: (BS, Frames, H, W, D)
        
    """
    
    psf_b, psf_c, psf_d, psf_h, psf_w = psf_vols.shape
    pad_zyx = [psf_d//2, psf_h//2, psf_w//2]
    #add padding to z, y, x 
    
    z = z + pad_zyx[0]
    y = y + pad_zyx[1]
    x = x + pad_zyx[2]

    #create padded tensor (bs, frame, c, h, w) We will need pad_size * 2 since we are padding from both sides
    loc3d_like = torch.zeros(output_shape[0], 
                             output_shape[1], 
                             output_shape[2] + 2*(pad_zyx[0]), 
                             output_shape[3] + 2*(pad_zyx[1]), 
                             output_shape[4] + 2*(pad_zyx[2])).to(x.device)
    
    if psf_c == 2:
        psf_ch_ind = torch.where(ch >= 8, 1, 0)
        psf_vols = psf_vols[torch.arange(len(psf_ch_ind)),psf_ch_ind]
    if output_shape[1] == 1:
        ch = torch.zeros_like(ch)
        
    psf_vols = psf_vols.reshape(-1, psf_d, psf_h, psf_w)
    
    # Take limit calculation out of the loop for 30% speed up
    z_l = z - pad_zyx[0]
    y_l = y - pad_zyx[1]
    x_l = x - pad_zyx[2]
    
    z_h = z + pad_zyx[0] + 1
    y_h = y + pad_zyx[1] + 1
    x_h = x + pad_zyx[2] + 1
    
    for idx in range(x.shape[0]):
        loc3d_like[b[idx], ch[idx],
        z_l[idx] : z_h[idx],
        y_l[idx] : y_h[idx],
        x_l[idx] : x_h[idx]] += psf_vols[idx]

    # unpad to original size
    b_sz, ch_sz, h_sz, w_sz, d_sz = loc3d_like.shape
    
    placed_psf = loc3d_like[:, :, pad_zyx[0]: h_sz - pad_zyx[0],
                                  pad_zyx[1]: w_sz - pad_zyx[1],
                                  pad_zyx[2]: d_sz - pad_zyx[2]]
    return placed_psf

def place_psf(locations, psf_volume, output_shape, device):
    """
    Places point spread functions (psf_volume) in to corresponding locations.

    Args:
        locations: tuple with the 5D voxel coordinates
        psf_volume: torch.Tensor
        output_shape: Shape Tuple(BS, C, H, W, D) 

    Returns:
        placed_psf: torch.Tensor with shape (BS, C, H, W, D)
    """

    b, c, z, y, x = locations
    
    if device == 'cpu':
        # Deprecated python loop
        placed_psf = _place_psf(psf_volume, b, c, z, y, x, torch.tensor(output_shape))
    
    if 'cuda' in device:
        # New fancy cuda loop
        N_psfs, _, psf_s_z, psf_s_y, psf_s_x = psf_volume.shape   
        placed_psf = CudaPlaceROI.apply(psf_volume[:,0], output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4], N_psfs, psf_s_z, psf_s_y, psf_s_x, b, c, z-psf_s_z//2, y-psf_s_y//2, x-psf_s_x//2)
    
#     assert placed_psf.shape == output_shape, f'output shape wrong {placed_psf.shape} != {output_shape}'
    
    return placed_psf

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

        from decode.emitter.process import RemoveOutOfFOV

        self._fov_filter = RemoveOutOfFOV(
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
        mask = self._fov_filter.clean_emitter(xyz)

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
        device: str = "cuda:0",
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

    """Pickle"""

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
        self, xyz: torch.Tensor, phot: torch.Tensor, bg: torch.Tensor, inversion=None
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
        self, xyz: torch.Tensor, phot: torch.Tensor, bg: torch.Tensor, inversion=None
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
    def __init__(
        self,
        xextent,
        yextent,
        zextent,
        *,
        weight_real: list,
        weight_im: list,
        theta: float,
        num_aperture: float,
        ref_index: float,
        wavelength: float,
        k_size: int,
    ):
        import zernike

        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent)

        self._weight_real = torch.tensor(weight_real)
        self._weight_im = torch.tensor(weight_im)
        self._theta = theta
        self._num_aperture = num_aperture
        self._ref_index = ref_index
        self._wavelength = wavelength
        self._k_size = k_size

        self._pupil_e_field = None

    @staticmethod
    def pupile_field(
        num_aperture: float,
        na: float,
        wavelength: float,
        na_sample: float,
        na_oil: float,
        na_cover: float,
        k_size: int,
        pupilradius: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute eletric field in pupile space

        Args:
            num_aperture: numerical aperture
            na: refractive index
            wavelength:
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
        kr = np.lib.scimath.sqrt(xx**2 + yy**2)
        kz = np.lib.scimath.sqrt(
            (na_oil / wavelength) ** 2 - (kr * na / wavelength) ** 2
        )

        kr = np.lib.scimath.sqrt(xx**2 + yy**2)
        kz = np.lib.scimath.sqrt(
            (na_oil / wavelength) ** 2 - (kr * na / wavelength) ** 2
        )

        cos_imm = np.lib.scimath.sqrt(1 - (kr * na / na_oil) ** 2)
        cos_med = np.lib.scimath.sqrt(1 - (kr * na / na_sample) ** 2)
        cos_cov = np.lib.scimath.sqrt(1 - (kr * na / na_cover) ** 2)

        kz_med = na_sample / wavelength * cos_med

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

    @staticmethod
    def zernike_space(n):
        """Calculate zernike polynomials"""

        ef = torch.cat([ef_x, ef_y], 0)
        return

    def _forward_single_frame(self, xyz: torch.Tensor, weight: torch.Tensor):
        h = self.t_a * self.zernike_real * torch.exp(self.zernike_im)
        h_k = h * self.e_mn * torch.exp(2j * math.pi * (1))
        real = self.fft(h_k) * weight

        return real

class VoxelatedPSF(PSF):

    def __init__(
        self,
        psf_cube: torch.tensor,     
        img_shape: Tuple[int, int, int],
        ref0: tuple,
        vx_size: Tuple[int, int, int],
        slice_mode: bool,
        n_channels: int,
        interpol_mode: str = 'bilinear',
        roi_size: (None, tuple) = None,
        ref_re: (None, torch.Tensor, tuple) = None,
        roi_auto_center: bool = False,
        device: str = "cuda:0",
    ):
        """
        Cubic spline PSF, i.e. a step-wise polynomial with coefficients of size X,Y,Z,64.

        Args:
            ref_re:
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            zextent (tuple): extent in z
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
            xextent=None, yextent=None, zextent=None, img_shape=img_shape
        )
        
        if psf_cube.ndim == 3:
            psf_cube = psf_cube.unsqueeze(0)
            
        self.psf_cube = psf_cube.to(device)
        self.n_colors = psf_cube.shape[0]
        
        self.n_channels = n_channels

        self.roi_size_px = (
            psf_cube.shape[1:]
        )
        
        self.slice_mode = slice_mode
        if self.slice_mode:
            assert img_shape[0] == 1, 'Slice data should be 1 dimensional in z'
        
        self.interpol_mode = interpol_mode
        assert self.interpol_mode in ['trilinear', 'bicubic'], 'Currently suppoers trilinear and bicubic interpolation'
        
        if self.interpol_mode == 'trilinear':
            self._intepol_impl = TrilinearInterpolation(self.psf_cube, self.slice_mode, device)
            
        if self.interpol_mode == 'bicubic':
            self._intepol_impl = BicubicInterpolation(self.psf_cube, self.slice_mode, device)

        if vx_size is None:
            vx_size = torch.Tensor([1.0, 1.0, 1.0])

        self.vx_size = (
            vx_size if isinstance(vx_size, torch.Tensor) else torch.Tensor(vx_size)
        )
        self.ref0 = ref0 if isinstance(ref0, torch.Tensor) else torch.Tensor(ref0)

        self._device, self._device_ix = decode.utils.hardware._specific_device_by_str(
            device
        )

    @property
    def _roi_size_nm(self):
        roi_size_nm = (
            self.roi_size_px[0] * self.vx_size[0],
            self.roi_size_px[1] * self.vx_size[1],
        )

        return roi_size_nm

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
        xyz_r[:, 2] = (xyz_r[:, 2] / self.vx_size[2]) % 1

        # place emitters according to the Reference (reference is in px)
        xyz_r = (xyz_r + self.ref0) * self.vx_size
        xyz_px = (
            (xyz_nm / self.vx_size - self.ref0)
            .floor()
            .int()
        )
        
        z_inds = None
        
        if self.slice_mode:
            
            z_val = xyz_nm[:,2]
            
            z_val = torch.clamp(0.5*z_val,-0.49999,0.49999) + 0.5 # transform to [0,1]
            z_scaled = z_val * (self.roi_size_px[0] - 2) # [0, z_size]
            z_inds = (torch.div(z_scaled, 1, rounding_mode='trunc')).type(torch.cuda.LongTensor) + 1
            
            xyz_r[:, 2] = -(z_scaled%1.) + 0.5 
            xyz_px[:, 2] = 0

        return xyz_r, xyz_px, z_inds

    def forward(
        self,
        xyz: torch.Tensor,
        weight: torch.Tensor,
        col_ix: Optional[torch.Tensor] = None,
        frame_ix: Optional[torch.Tensor] = None,
        ch_ix: Optional[torch.Tensor] = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ):
        """
        Forward coordinates frame index aware through the psf model.

        Args:
            xyz: coordinates of size N x (2 or 3)
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

        xyz_r, xyz_ix, z_slice_ix = self.frame2roi_coord(xyz)

        n_frames = ix_high - ix_low
        frames_shape = [n_frames, self.n_channels] + self.img_shape
        
        shifted_psfs = self._intepol_impl.forward(xyz_r, z_slice_ix, col_ix)
        
        shifted_psfs = shifted_psfs * weight.view(-1, 1, 1, 1, 1).to(self._device)
        
        frames = place_psf([frame_ix.to(torch.long).to(self._device), 
                            ch_ix.to(torch.long).to(self._device), 
                            xyz_ix[:,2].to(self._device), 
                            xyz_ix[:,1].to(self._device), 
                            xyz_ix[:,0].to(self._device)], 
                            shifted_psfs, frames_shape, self._device)

        return frames