import math
import os
import sys
from abc import ABC, abstractmethod  # abstract class

import numpy as np
import scipy
from scipy.stats import binned_statistic_2d
import torch
from torch.autograd import Function

import deepsmlm.generic.emitter
import torch_cpp as tp


class PSF(ABC):
    """
    Abstract class to represent a point spread function.
    __init__ and forward must be overwritten.
    """

    @abstractmethod
    def __init__(self,
                 xextent=(None, None),
                 yextent=(None, None),
                 zextent=None,
                 img_shape=(None, None)):
        """
        Note: xextent should include the complete pixel area/volume, i.e. not
        only the px midpoint coordinates. If we have two 1D "pixels", which midpoints
        are to be at 0 and 1, then xextent is (-.5, 1.5)

        :param xextent: extent in x in px
        :param yextent: extent in y in px
        :param zextent: extent in z in px / voxel (not in nm) or None
        :param img_shape: shape of the image, tuple of 2 or 3 elements (2D / 3D)
        """

        ABC.__init__(self)
        Function.__init__(self)

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.img_shape = img_shape

    @abstractmethod
    def forward(self, input, weight):
        """
        Abstract method to go from position-matrix and photon number (aka weight) to an image.
        Call it before implementing your psf to be able to parse deepsmlm.generic.emitter.EmitterSets (super().forward(x, weight)).
        """
        if isinstance(input, deepsmlm.generic.emitter.EmitterSet):
            pos = input.xyz
            if weight is None:
                weight = input.phot

            return pos, weight

        else:
            return input, weight

    def __str__(self):
        return 'PSF: \n xextent: {}\n yextent: {}\n zextent: {}\n img_shape: {}'.format(self.xextent,
                                                                                       self.yextent,
                                                                                       self.zextent,
                                                                                       self.img_shape)

    def print_basic_properties(self):
        print(self)


class DeltaPSF(PSF):
    """
    Delta function PSF. You input a list of coordinates,
    psf forwards an image where a single non-zero px corresponds to an emitter.
    """

    def __init__(self, xextent, yextent, zextent, img_shape,
                 photon_threshold=None,
                 photon_normalise=False,
                 dark_value=None,
                 ambigous_default='max'):
        """
        (See abstract class constructor.)
        :param photon_normalise: normalised photon count, i.e. probabilities
        :param dark_value: Value where there is no emitter. Usually 0, but might be non-zero if used for a mask.
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape)
        self.photon_threshold = photon_threshold
        self.photon_normalise = photon_normalise
        self.dark_value = dark_value
        """
        Binning in numpy: binning is (left Bin, right Bin]
        (open left edge, including right edge)
        """
        self.bin_x = np.linspace(xextent[0], xextent[1],
                                 img_shape[0] + 1, endpoint=True)
        self.bin_y = np.linspace(yextent[0], yextent[1],
                                 img_shape[1] + 1, endpoint=True)

        if self.zextent is not None:
            raise DeprecationWarning("Not tested and not developed further.")

    def forward(self, input, weight=None):
        """

        :param input:  instance of emitterset or coordinates.
        :param weight:  number of photons or any other 1:1 connection to an emitter

        :return:  torch tensor of size 1 x H x W
        """
        def max_0(values):
            if values.__len__() == 0:
                return 0
            else:
                return np.max(values)

        xyz, weight = super().forward(input, weight)

        if self.photon_threshold is not None:
            raise DeprecationWarning("Not supported anymore. Use a fresh target generator and put it into a sequence.")

        if self.photon_normalise:
            weight = torch.ones_like(weight)

        # camera, _, _ = np.histogram2d(xyz[:, 0].numpy(), xyz[:, 1].numpy(),  # reverse order
        #                               bins=(self.bin_x, self.bin_y),
        #                               weights=weight.numpy())
        if xyz.size(0) == 0:
            camera = torch.zeros(self.img_shape).numpy()
        else:
            camera, _, _, _ = binned_statistic_2d(xyz[:, 0].numpy(), xyz[:, 1].numpy(), weight.numpy(),
                                                  bins=(self.bin_x, self.bin_y), statistic=max_0)

        camera = torch.from_numpy(camera.astype(np.float32)).unsqueeze(0)
        if self.dark_value is not None:
            camera[camera == 0.] = self.dark_value

        return camera


class OffsetPSF(DeltaPSF):
    """
    Coordinate to px-offset class.
    """
    def __init__(self, xextent, yextent, img_shape):
        super().__init__(xextent, yextent, None, img_shape,
                         photon_threshold=0,
                         photon_normalise=False,
                         dark_value=0.)

        """Setup the bin centers x and y"""
        self.bin_x = torch.from_numpy(self.bin_x).type(torch.float)
        self.bin_y = torch.from_numpy(self.bin_y).type(torch.float)
        self.bin_ctr_x = (0.5 * (self.bin_x[1] + self.bin_x[0]) - self.bin_x[0] + self.bin_x)[:-1]
        self.bin_ctr_y = (0.5 * (self.bin_y[1] + self.bin_y[0]) - self.bin_y[0] + self.bin_y)[:-1]

        self.offset_max_x = self.bin_x[1] - self.bin_ctr_x[0]
        self.offset_max_y = self.bin_y[1] - self.bin_ctr_y[0]

    def forward(self, em):
        """
        :param emitter: emitterset
        :return: (torch.tensor), dim: 2 x img_shape[0] x img_shape[1] ("C x H x W"),
        where C=0 is the x offset and C=1 is the y offset.
        """

        xy_offset_map = torch.zeros((2, *self.img_shape))
        # loop over all emitter positions
        for i in range(len(em)):
            xy = em.xyz[i, :2]
            """
            If position is outside the FoV, skip.
            Find ix of px in bin. bins must be sorted. Remember that in numpy bins are (a, b].
            (from inner to outer). 1. get logical index of bins, 2. get nonzero where condition applies, 
            3. use the min value
            """
            if xy[0] > self.bin_x.max() or xy[0] <= self.bin_x.min() \
                    or xy[1] > self.bin_y.max() or xy[1] <= self.bin_y.min():
                continue

            x_ix = (xy[0].item() > self.bin_x).nonzero().max(0)[0].item()
            y_ix = (xy[1].item() > self.bin_y).nonzero().max(0)[0].item()
            xy_offset_map[0, x_ix, y_ix] = xy[0] - self.bin_ctr_x[x_ix]  # coordinate - midpoint
            xy_offset_map[1, x_ix, y_ix] = xy[1] - self.bin_ctr_y[y_ix]  # coordinate - midpoint

        return xy_offset_map


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

    def forward(self, input, weight=None):
        """
        Forward emitters through PSF. Note that this ignores the frame_ix of a possible emitterset (so it puts all
        emitters on the same frame). If you want different behaviour, split the emitters in frames ([
        ].split_in_frames()) and loop over the frames or use simulator to do this for you.

        Args:
            input: instance of emitterset or xyz coordinates
            weight: override weight of emitterset (otherwise uses photon count) or specify weight if xyz were given

        Returns:
            single camera frame
        """

        pos, weight = super().forward(input, weight)
        num_emitters = pos.shape[0]
        img_shape = self.img_shape
        sigma_0 = self.sigma_0

        if num_emitters == 0:
            return torch.zeros(1, img_shape[0], img_shape[1]).float()

        xpos = pos[:, 0].repeat(img_shape[0], img_shape[1], 1)
        ypos = pos[:, 1].repeat(img_shape[0], img_shape[1], 1)

        if self.zextent is not None:
            sig = self.astigmatism(pos[:, 2], sigma_0=self.sigma_0)
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

        return gaussCdf.unsqueeze(0)


class GPUSplinePSF(PSF):

    def __init__(self, xextent, yextent, img_shape, roi_size, coeff, vx_size, ref0, frame_mode='abs',
                 frame_ref=0, cuda=True):
        """
        Initialise Spline PSF

        Args:
            roi_size:
            coeff:
            vx_size: pixel / voxel size
            ref0: zero reference point (in px / vx units)
            cuda:
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=None, img_shape=img_shape)
        import spline_psf_cuda

        self.roi_size_px = roi_size
        self._coeff = coeff
        self.vx_size = vx_size
        self.ref0 = ref0
        self.cuda = cuda
        self.frame_mode = frame_mode
        self.frame_ref = frame_ref

        self._spline_cuda = spline_psf_cuda.PSFWrapperCUDA(coeff.shape[0], coeff.shape[1], coeff.shape[2], roi_size[0],
                                                           roi_size[1], coeff.numpy())

        self._spline_cpu = spline_psf_cuda.PSFWrapperCPU(coeff.shape[0], coeff.shape[1], coeff.shape[2], roi_size[0],
                                                         roi_size[1], coeff.numpy())

        self._safety_check()

    @property
    def _roi_size_nm(self):
        roi_size_nm = (self.roi_size_px[0] * self.vx_size[0],
                       self.roi_size_px[1] * self.vx_size[1])

        return roi_size_nm

    def _safety_check(self):
        """
        Perform some class specific safety checks
        Returns:

        """
        """Test whether extent corresponds to img shape"""
        if (self.img_shape[0] != (self.xextent[1] - self.xextent[0])) or \
                (self.img_shape[1] != (self.yextent[1] - self.yextent[0])):
            raise ValueError("Unequal size of extent and image shape not supported.")

    def nm2impl(self, xyz_nm):
        """
        Transforms nanometre coordinates to implementation coordiantes

        Args:
            xyz_nm: (torch.Tensor)

        Returns:

        """
        return -xyz_nm/self.vx_size + self.ref0

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
        """Place emitters in ROI centre (nm)"""
        xyz_r[:, :2] = (xyz_r[:, :2] + self.ref0[:2]) * self.vx_size[:2]

        xyz_px = (xyz_nm[:, :2] / self.vx_size[:2] - self.ref0[:2]).floor().int()

        return xyz_r, xyz_px

    def forward_rois(self, xyz, phot):
        return self._forward_rois_impl(self.nm2impl(xyz), phot)

    def forward(self, xyz, phot, frame_ix):
        xyz_ = self.nm2impl(self.frame2roi_coord(xyz))
        return self._forward_impl(self.nm2impl)

    def _forward_rois_impl(self, xyz, phot):
        """
        Computes the PSF and outputs the result ROI-wise.

        Args:
            xyz:
            phot:

        Returns:

        """

        n_rois = xyz.size(0)  # number of rois / emitters / fluorophores

        if self.cuda:
            out = self._spline_cuda.forward_rois(xyz[:, 0], xyz[:, 1], xyz[:, 2], phot)
        else:
            out = self._spline_cpu.forward_rois(xyz[:, 0], xyz[:, 1], xyz[:, 2], phot)

        out = torch.from_numpy(out)
        out = out.reshape(n_rois, *self.roi_size_px)
        return out

    def _place_rois(self, rois, ix, frame_ix, n_frames):
        raise DeprecationWarning("Was only for implementation.")
        frame = torch.zeros((n_frames, *self.img_shape))

        for r in range(rois.size(0)):
            for i in range(self.roi_size_px[0]):
                for j in range(self.roi_size_px[1]):
                    ii = ix[r, 0] + i
                    jj = ix[r, 1] + j
                    if (ii < 0) or (jj < 0) or (ii >= self.img_shape[0]) or (jj >= self.img_shape[1]):
                        continue
                    frame[frame_ix[r], ii, jj] += rois[r, i, j]

        return frame

    def forward(self, xyz, phot, frame_ix):
        """

        Args:
            xyz:
            phot:
            frame_ix:

        Returns:

        """
        """Convert Coordinates into ROI based coordinates and transform into implementation coordinates"""
        xyz_r, ix = self.frame2roi_coord(xyz)
        xyz_r = self.nm2impl(xyz_r)

        # Calc number of frames
        if self.frame_mode == 'abs':
            n_frames = (frame_ix.max() - self.frame_ref + 1).int().item()
            frame_ix_ = frame_ix

        elif self.frame_mode == 'rel':
            n_frames = (frame_ix.max() - frame_ix.min() + 1).int().item()
            frame_ix_ = frame_ix - frame_ix.min()

        if n_frames == 0:
            return torch.zeros((0, *self.img_shape))

        frames = self._spline_cpu.forward_frames(*self.img_shape,
                                                 frame_ix_,
                                                 n_frames,
                                                 xyz_r[:, 0],
                                                 xyz_r[:, 1],
                                                 xyz_r[:, 2],
                                                 ix[:, 0],
                                                 ix[:, 1],
                                                 phot)
        frames = torch.from_numpy(frames).reshape(n_frames, *self.img_shape)
        return frames


class CubicSplinePSF(PSF):
    """
    Cubic spline psf. This is the PSF of use for simulation.
    """

    native_order = 'xypbz'
    max_factor_nat_order = torch.tensor([10., 10., 10, 10, 10.]).unsqueeze(0) ** 2
    max_value_nat_order = torch.tensor([3., 3., 10000., 200., 1000.]) ** 2
    big_number_handling = 'max_value'
    n_par = 5

    def __init__(self, xextent, yextent, zextent, img_shape, coeff, ref0, dz=None, crlb_order='xyzpb'):
        """
        (see abstract class constructor

        :param coeff: coefficient matrix / tensor of the cubic spline. Hc x Wc x Dc x 64 (Hc, Wc, DC height, width,
                        depth with which spline was fitted)
        :param ref0: index relative to coefficient matrix which gives the "midpoint of the psf"
        :param dz: distance between z slices. You must provide either zextent or dz. If both, dz will be used.
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape)
        if img_shape[0] != img_shape[1]:
            raise ValueError("Image must be of equal size in x and y.")
        self.npx = img_shape[0]

        self.coeff = coeff
        self.ref0 = ref0
        if dz is None:  # if dz is None, zextent must not be None
            if zextent is None:
                raise ValueError('Either you must provide zextent or you must provide dz.')
            dz = (self.zextent[1] - self.zextent[0]) / (self.coeff.shape[2] - 1)

        self.dz = dz
        self.crlb_order = crlb_order

        self.spline_c = tp.initSpline(self.coeff.type(torch.FloatTensor),
                                       list(self.ref0),
                                       self.dz)

        self.roi_size = (self.coeff.size(0), self.coeff.size(1), self.coeff.size(2))

        """Test whether extent corresponds to img shape"""
        if (img_shape[0] != (xextent[1] - xextent[0])) or (img_shape[1] != (yextent[1] - yextent[0])):
            raise ValueError("Unequal size of extent and image shape not supported.")

    def print_basic_properties(self):
        super().print_basic_properties()
        print(f' ROI size: {self.roi_size}')

    def f(self, x, y, z):
        return tp.f_spline(self.spline_c, x, y, z)

    def d(self, pos, phot, bg):

        return tp.f_spline_d(self.spline_c, pos, phot, bg, self.npx, list((self.xextent[0], self.yextent[0])))

    def fisher(self, pos, phot, bg):
        """Outputs the Fisher matrix in CPP order"""

        fisher = tp.f_spline_fisher(self.spline_c, pos, phot, bg, self.npx, list((self.xextent[0], self.yextent[0])))
        # ToDo: warning because z in the fisher is not in nm but in multiples of dz
        return fisher

    def _rearange_crlb(self, cr, crlb_order=None):
        """Outputs the CRLB"""
        if crlb_order == self.native_order:
            return cr
        elif crlb_order == 'xyzpb':
            if cr.dim() == 1:
                cr = cr[[0, 1, 4, 2, 3]]
            else:
                cr = cr[:, [0, 1, 4, 2, 3]]
            return cr
        else:
            cr_dict = dict()
            cr_dict['x'] = cr[0]
            cr_dict['y'] = cr[1]
            cr_dict['z'] = cr[4]
            cr_dict['phot'] = cr[2]
            cr_dict['bg'] = cr[3]
            return cr_dict

    def crlb_single(self, pos, phot, bg, crlb_order=None):
        """
        Computes the cramer rao lower bound as if the emitters were isolated

        Args:
            pos: torch.Tensor, N x 3
            phot: torch.Tensor, N
            bg: torch.Tensor, N
            crlb_order: (optional) order of the output

        Returns:
            cr: cramer rao bound on pos, phot, bg in the order specified
            img: calculated frae, since this is for free

        """

        cr = []
        img = []

        """Just call the standard crlb calc method but split the inputs."""
        n_emitters = pos.size(0)

        if n_emitters == 0:
            cr = torch.zeros((0, 5))
            img = torch.zeros(self.img_shape)

            return cr, img

        for i in range(n_emitters):
            cr_, img_ = self.crlb(pos=pos[[i], :], phot=phot[[i]], bg=bg[[i]], crlb_order=crlb_order)
            cr.append(cr_)
            img.append(img_)

        # Put things together. Stack the cr values, the img may be added.
        cr = torch.cat(cr, dim=0)
        img = torch.stack(img, dim=img[0].dim()).sum(-1)

        return cr, img

    def crlb(self, pos, phot, bg, crlb_order=None):
        """
           Computes the cramer rao lower bound

           Args:
                pos: torch.Tensor, N x 3
                phot: torch.Tensor, N
                bg: torch.Tensor, N
                crlb_order: (optional) order of the output

            Returns:
                cr: cramer rao bound on pos, phot, bg in the order specified
                img: calculated frae, since this is for free

           """
        if crlb_order is None:
            crlb_order = self.crlb_order

        # fisher, img = self.fisher(pos, phot, bg)
        # cr = fisher.pinverse().diag().view(pos.size(0), -1)
        cr, img = tp.f_spline_crlb(self.spline_c, pos, phot, bg, self.npx, list((self.xextent[0], self.yextent[0])))

        # rescale the cr in z by dz because we input z in nm not in multiples of dz.
        cr[:, 4] *= self.dz**2

        cr = self._rearange_crlb(cr, crlb_order)
        cr[torch.isnan(cr)] = float('inf')

        """
        NaN and big number handling.
        """
        if self.big_number_handling == 'max_value':
            max_val_tensor = self._rearange_crlb(self.max_value_nat_order, crlb_order).unsqueeze(0)
            max_val_tensor = max_val_tensor.repeat(cr.size(0), 1)
            is_to_big = cr >= max_val_tensor
            cr[is_to_big] = max_val_tensor[is_to_big]
            cr[cr < 0] = max_val_tensor[cr < 0]

        elif self.big_number_handling == 'single_crlb':
            """
            Clamp and NaN Handling. Calculate the crlb as if the emitter was not surrounded by others. The Multi-CRLB shall
            then not exceed the individual crlb by a given factor. NaNs are handled by single_crlb * max_factor
            """
            crlb_ind, _ = self.crlb_single(pos, phot, bg, crlb_order, fisher, img)

            # clamp by the max values
            max_factor_ordered = self._rearange_crlb(self.max_factor_nat_order, crlb_order)
            max_crlb = crlb_ind * max_factor_ordered
            cr[cr > max_crlb] = max_crlb[cr > max_crlb]

        elif self.big_number_handling is None:
            pass

        else:
            raise ValueError("Not supported big number handling in CRLB calculation.")

        return cr, img

    def forward(self, pos, weight=None):
        pos, weight = super().forward(pos, weight)

        if (pos is not None) and (pos.shape[0] != 0):
            return tp.fPSF(self.spline_c,
                           pos.type(torch.FloatTensor),
                           weight.type(torch.FloatTensor),
                           self.npx,
                           list((self.xextent[0], self.yextent[0])))
        else:
            return torch.zeros((1, self.img_shape[0], self.img_shape[1])).type(torch.FloatTensor)


class GaussianSampleBased(PSF):
    """
    Gold standard on how to draw samples from a distribution
    """

    def __init__(self, xextent, yextent, zextent, img_shape, sigma_0):
        """
        (See abstract class constructor.)

        :param sigma_0:   intial sigma.
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape)

        self.sigma_0 = sigma_0

    def forward(self, pos, weight):
        raise NotImplementedError(
            "Not implemented and useable at the moment. Needs the ability for multiple emitters.")

        mu = pos[:2]
        cov = np.power(np.array([[sig[0], 0], [0, sig[1]]]), 2)
        cov, _, _ = astigmatism(pos, cov)
        phot_pos = np.random.multivariate_normal(mu, cov, int(photon_count))  # in nm

        shape = img_shape
        # extent of our coordinate system
        if xextent is None:
            xextent = np.array([0, shape[0]], dtype=float)
        if yextent is None:
            yextent = np.array([0, shape[1]], dtype=float)

        if origin == 'px_centre':  # shift 0 right towards the centre of the first px, and down
            xextent -= 0.5
            yextent -= 0.5
        '''
        Binning in numpy: binning is (left Bin, right Bin]
        (open left edge, including right edge)
        '''
        bin_rows = np.linspace(xextent[0], xextent[1], img_shape[0] + 1, endpoint=True)
        bin_cols = np.linspace(yextent[0], yextent[1], img_shape[1] + 1, endpoint=True)

        camera, xedges, yedges = np.histogram2d(phot_pos[:, 1], phot_pos[:, 0], bins=(
            bin_rows, bin_cols))  # bin into 2d histogram with px edges

        raise NotImplementedError('Image vector has wrong dimensionality. Need to add singleton dimension for channels.')

        return camera


class ListPseudoPSF(PSF):
    def __init__(self, xextent, yextent, zextent, dim=3, photon_threshold=0):
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=None)
        self.dim = dim
        self.photon_threshold = photon_threshold

    def forward(self, emitter):
        pos, weight = emitter.xyz, emitter.phot

        """Threshold the photons."""
        ix = weight > self.photon_threshold
        pos = pos[ix, :]
        weight = weight[ix]

        if self.dim == 3:
            return pos[:, :3], weight
        elif self.dim == 2:
            return pos[:, :2], weight
        else:
            raise ValueError("Wrong dimension.")


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import os

    import deepsmlm.generic.psf_kernel
    import deepsmlm.generic.plotting.frame_coord as fc
    import deepsmlm.generic.inout.load_calibration

    deepsmlm_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     os.pardir, os.pardir)) + '/'

    coeff_file = deepsmlm_root + 'data_central/Calibration/2019/M2_CollabSpeiser/000_beads_640i100_x35_Z-stack_1_MMStack_Pos0.ome_3dcal.mat'
    smap_load = deepsmlm.generic.inout.load_calibration.SMAPSplineCoefficient(coeff_file)
    coeff = smap_load.coeff

    psf_cu = deepsmlm.generic.psf_kernel.GPUSplinePSF((-0.5, 63.5), (-0.5, 63.5), (64, 64), (32, 32),
                                                      coeff.contiguous(), torch.Tensor([100., 100., 10.]),
                                                      torch.Tensor([13, 13, 100]), cuda=True)

    psf_cpu = deepsmlm.generic.psf_kernel.GPUSplinePSF((-0.5, 63.5), (-0.5, 63.5), (64, 64), (32, 32),
                                                      coeff.contiguous(), torch.Tensor([100., 100., 10.]),
                                                      torch.Tensor([13, 13, 100]), cuda=False)

    """RUNTIME TEST"""
    n = 100
    print(f"Num samples: {n}")
    xyz_plain = torch.rand((n, 3))
    xyz_plain[:, :2] *= 5000
    xyz_plain[:, 2] *= 1000
    xyz_plain[:, 2] -= 500
    phot = torch.ones((n, ))
    frame_ix = torch.zeros_like(phot).int() - 1
    frame_ix[-1] = 0
    # frame_ix = torch.randint(0, 10, size=(n, ))

    frames_cpu = psf_cpu.forward(xyz_plain, phot, frame_ix)
    fc.PlotFrame(frames_cpu[0]).plot()
    plt.show()



    print("Done.")




