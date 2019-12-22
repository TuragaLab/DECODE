import math
import os
import sys
from abc import ABC, abstractmethod  # abstract class

import numpy as np
import scipy
from scipy.stats import binned_statistic_2d
import torch
from torch.autograd import Function
from deepsmlm.generic.emitter import EmitterSet
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
        Call it before implementing your psf to be able to parse emittersets (super().forward(x, weight)).
        """
        if isinstance(input, EmitterSet):
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
        for i in range(em.num_emitter):
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
        (See abstract class constructor.)

        :param sigma_0: initial sigma value in px dimension
        :param peak_weight: use weight for determining the peak intensity instead of
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape)

        self.sigma_0 = sigma_0
        self.peak_weight = peak_weight

    def astigmatism(self, z, sigma_0, m=torch.tensor([0.004, 0.001])):
        """
        Astigmatism function for gaussians which are 3D.

        :param z_nm: z value in nanometer

        :return: sigma values
        """

        # raise NotImplementedError
        sigma_xy = sigma_0 * torch.ones(z.shape[0], 2)  # change behaviour as this is tuple

        sigma_xy[z > 0, :] = torch.cat(
            ((sigma_0 + m[0] * z[z > 0].abs()).unsqueeze(1),
             (sigma_0 + m[1] * z[z > 0].abs()).unsqueeze(1)), 1)

        sigma_xy[z < 0, :] = torch.cat(
            ((sigma_0 + m[1] * z[z < 0].abs()).unsqueeze(1),
             (sigma_0 + m[0] * z[z < 0].abs()).unsqueeze(1)), 1)

        return sigma_xy

    def forward(self, input, weight):
        """

        :param pos:  position of the emitter in 2 or 3D
        :param weight:  number of photons or any other 1:1 connection to an emitter
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


class SplineCPP(PSF):
    """
    Spline Function wrapper for C++ / C
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

        fisher, img = self.fisher(pos, phot, bg)
        cr = fisher.pinverse().diag().view(pos.size(0), -1)
        # cr, img = tp.f_spline_crlb(self.spline_c, pos, phot, bg, self.npx, list((self.xextent[0], self.yextent[0])))

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