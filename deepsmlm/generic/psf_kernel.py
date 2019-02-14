from abc import ABC, abstractmethod  # abstract class
import numpy as np
import math
import torch
from torch.autograd import Function


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
    def forward(self, pos, weight):
        """
        Abstract method to go from position-matrix and photon number (aka weight) to an image.
        """
        pass


class DeltaPSF(PSF):
    """
    Delta function PSF. You input a list of coordinates,
    psf forwards an image where a single non-zero px corresponds to an emitter.
    """

    def __init__(self, xextent, yextent, zextent, img_shape):
        """
        (See abstract class constructor.)
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape)

        """
        Binning in numpy: binning is (left Bin, right Bin]
        (open left edge, including right edge)
        """
        self.bin_x = np.linspace(xextent[0], xextent[1],
                                 img_shape[0] + 1, endpoint=True)
        self.bin_y = np.linspace(yextent[0], yextent[1],
                                 img_shape[1] + 1, endpoint=True)

        if self.zextent is not None:
            self.bin_z = np.linspace(zextent[0], zextent[1],
                                     img_shape[2] + 1, endpoint=True)

    def forward(self, pos, weight):
        """

        :param pos:  position of the emitter in 2 or 3D
        :param weight:  number of photons or any other 1:1 connection to an emitter

        :return:  torch tensor of size 1 x H x W
        """

        if self.zextent is None:
            camera, _, _ = np.histogram2d(pos[:, 0].numpy(), pos[:, 1].numpy(),  # reverse order
                                          bins=(self.bin_x, self.bin_y),
                                          weights=weight.numpy())
        else:
            camera, _ = np.histogramdd((pos[:, 0].numpy(), pos[:, 1].numpy(), pos[:, 2].numpy()),
                                       bins=(self.bin_x, self.bin_z, self.bin_z),
                                       weights=weight.numpy())

        return torch.from_numpy(camera.astype(np.float32)).unsqueeze(0)


class DualDelta(DeltaPSF):
    """
    Delta function PSF in channel 0: photons, channel 1 z position.
    Derived from DeltaPSF class.
    """

    def __init__(self, xextent, yextent, zextent, img_shape):
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape)

    def forward(self, pos, weight, weight2):
        """

        :param pos: position of the emitter in 2 or 3D
        :param weight:  number of photons or any other 1:1 connection to an emitter
        :param weight2: z position or any other 1:1 connection to an emitter

        :return: torch tensor of size 2 x H x W
        """
        dual_ch_img = torch.cat((
            super(DualDelta, self).forward(pos, weight),
            super(DualDelta, self).forward(pos, weight2)),
            dim=0)

        return dual_ch_img


class GaussianExpect(PSF):
    """
    A gaussian PSF which models the by using the function values of the probability distribution.
    You must not use this function without shotnoise for simulation.
    """

    def __init__(self, xextent, yextent, zextent, img_shape, sigma_0):
        """
        (See abstract class constructor.)

        :param sigma_0: initial sigma value in px dimension
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape)

        self.sigma_0 = sigma_0

    def astigmatism(self, z, sigma_0, m=torch.tensor([0.4, 0.1])):
        """
        Astigmatism function for gaussians which are 3D.

        :param z_nm: z value in nanometer

        :return: sigma values
        """

        raise NotImplementedError
        sigma_xy = sigma_0 * torch.ones(znm.shape[0], 2)  # change behaviour as this is tuple

        sigma_xy[z > 0, :] *= torch.cat(
            ((1 + m[0] * z[z > 0]).unsqueeze(1), (1 + m[1] * z[z > 0]).unsqueeze(1)), 1)

        sigma_xy[z < 0, :] *= torch.cat(
            ((1 - m[0] * z[z < 0]).unsqueeze(1), (1 - m[1] * z[z < 0]).unsqueeze(1)), 1)

        return sigma_xy

    def forward(self, pos, weight):
        """

        :param pos:  position of the emitter in 2 or 3D
        :param weight:  number of photons or any other 1:1 connection to an emitter
        """

        num_emitters = pos.shape[0]
        img_shape = self.img_shape
        sigma_0 = self.sigma_0

        if num_emitters == 0:
            return torch.zeros(1, img_shape[0], img_shape[1], dtype=torch.float32)

        xpos = pos[:, 0].repeat(img_shape[0], img_shape[1], 1)
        ypos = pos[:, 1].repeat(img_shape[0], img_shape[1], 1)

        if self.zextent is not None:
            sig = astigmatism(pos, sigma_0=self.sigma_0)
            sig_x = sig[:, 0].repeat(img_shape[0], img_shape[1], 1)
            sig_y = sig[:, 1].repeat(img_shape[0], img_shape[1], 1)
        else:
            sig_x = sigma_0[0]
            sig_y = sigma_0[1]

        x = torch.linspace(self.xextent[0], self.xextent[1], img_shape[0] + 1, dtype=torch.float32)
        y = torch.linspace(self.yextent[0], self.yextent[1], img_shape[1] + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y)

        xx = xx.unsqueeze(2).repeat(1, 1, num_emitters)
        yy = yy.unsqueeze(2).repeat(1, 1, num_emitters)

        print(xx.shape)

        gauss_x = torch.erf((xx[1:, 1:, :] - xpos) / (math.sqrt(2) * sig_x)) \
            - torch.erf((xx[0:-1, 1:, :] - xpos) / (math.sqrt(2) * sig_x))

        gauss_y = torch.erf((yy[1:, 1:, :] - ypos) / (math.sqrt(2) * sig_y)) \
            - torch.erf((yy[1:, 0:-1, :] - ypos) / (math.sqrt(2) * sig_y))

        gaussCdf = weight.type_as(gauss_x) / 4 * torch.mul(gauss_x, gauss_y)
        gaussCdf = torch.sum(gaussCdf, 2)

        return gaussCdf


class SplineExpect(PSF):
    """
    Partly based on
    https://github.com/ZhuangLab/storm-analysis/blob/master/storm_analysis/spliner/spline3D.py
    """

    def __init__(self, xextent, yextent, zextent, img_shape, coeff=None):
        """
        (See abstract class constructor.)

        :param coeff:   cubic spline coefficient matrix. dimension: Nx * Ny * Nz * 64
        """
        super().__init__(xextent=xextent, yextent=yextent, zextent=zextent, img_shape=img_shape)

        self.coeff = coeff
        self.max_i = torch.as_tensor(coeff.shape, dtype=torch.float32) - 1

    def roundAndCheck(self, x, max_x):
        if (x < 0.0) or (x > max_x):
            return [-1, -1]

        x_floor = torch.floor(x)
        x_diff = x - x_floor
        ix = int(x_floor)
        if (x == max_x):
            ix -= 1
            x_diff = 1.0

        return [ix, x_diff]

    def dxf(self, x, y, z):
        [ix, x_diff] = self.roundAndCheck(x, self.max_i[0])
        [iy, y_diff] = self.roundAndCheck(y, self.max_i[1])
        [iz, z_diff] = self.roundAndCheck(z, self.max_i[2])

        if (ix == -1) or (iy == -1) or (iz == -1):
            return 0.0

        yval = 0.0
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    yval += float(i+1) * self.coeff[ix, iy, iz, (i+1)*16+j*4+k] * torch.pow(x_diff, i) * torch.pow(y_diff, j) * torch.pow(z_diff, k)
        return yval

    def dyf(self, x, y, z):
        [ix, x_diff] = self.roundAndCheck(x, self.max_i[0])
        [iy, y_diff] = self.roundAndCheck(y, self.max_i[1])
        [iz, z_diff] = self.roundAndCheck(z, self.max_i[2])

        if (ix == -1) or (iy == -1) or (iz == -1):
            return 0.0

        yval = 0.0
        for i in range(4):
            for j in range(3):
                for k in range(4):
                    yval += float(j+1) * self.coeff[ix, iy, iz, i*16+(j+1)*4+k] * torch.pow(x_diff, i) * torch.pow(y_diff, j) * torch.pow(z_diff, k)
        return yval

    def dzf(self, x, y, z):
        [ix, x_diff] = self.roundAndCheck(x, self.max_i[0])
        [iy, y_diff] = self.roundAndCheck(y, self.max_i[1])
        [iz, z_diff] = self.roundAndCheck(z, self.max_i[2])

        if (ix == -1) or (iy == -1) or (iz == -1):
            return 0.0

        yval = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    yval += float(k+1) * self.coeff[ix, iy, iz, i*16+j*4+k+1] * torch.pow(x_diff, i) * torch.pow(y_diff, j) * torch.pow(z_diff, k)
        return yval

    def f(self, x, y, z):
        [ix, x_diff] = self.roundAndCheck(x, self.max_i[0])
        [iy, y_diff] = self.roundAndCheck(y, self.max_i[1])
        [iz, z_diff] = self.roundAndCheck(z, self.max_i[2])

        if (ix == -1) or (iy == -1) or (iz == -1):
            return 0.0

        f = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    f += self.coeff[ix, iy, iz, i * 16 + j * 4 + k] * \
                        torch.pow(x_diff, i) * torch.pow(y_diff, j) * \
                        torch.pow(z_diff, k)
        return f

    def forward(self, input, weight):
        raise NotImplementedError


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xextent = (-.5, 31.5)
    yextent = (-.5, 15.5)
    zextent = None
    img_shape = (32, 16)

    delta_psf = DeltaPsf(xextent, yextent, zextent, img_shape)
    gauss_psf = GaussianExpect(xextent, yextent, zextent, img_shape, sigma_0=(1.5, 1.5))

    xy = torch.rand((3, 2)) * 16
    xy = torch.tensor([[0., 0], [15, 2], [30, 10]], dtype=torch.float32)
    phot = torch.randint(0, 5000, (3, ))
    print(xy, phot)

    plt.subplot(121)
    plt.imshow(delta_psf.forward(xy, phot).squeeze())
    plt.subplot(122)
    plt.imshow(gauss_psf.forward(xy, phot).squeeze())

    plt.show()
