import numpy as np
import scipy.io as sio
import torch

import deepsmlm.generic.psf_kernel


class SMAPSplineCoefficient:
    """Wrapper class as an interface for MATLAB Spline calibration data."""
    def __init__(self, file, spline_obj=deepsmlm.generic.psf_kernel.CubicSplinePSF):
        """

        :param file: .mat file
        """
        self.calib_file = file
        self.calib_mat = sio.loadmat(self.calib_file, struct_as_record=False, squeeze_me=True)['SXY']

        self.coeff = torch.from_numpy(self.calib_mat.cspline.coeff)
        self.ref0 = (self.calib_mat.cspline.x0 - 1, self.calib_mat.cspline.x0 - 1, self.calib_mat.cspline.z0)
        self.dz = self.calib_mat.cspline.dz
        self.spline_roi_shape = self.coeff.shape[:3]
        self.spline_obj = spline_obj

    def init_spline(self, xextent, yextent, img_shape):
        """
        Initialise Spline
        :return: spline instance
        """
        psf = self.spline_obj(xextent=xextent,
                              yextent=yextent,
                              zextent=None,
                              img_shape=img_shape,
                              coeff=self.coeff,
                              ref0=self.ref0,
                              dz=self.dz)

        psf.print_basic_properties()
        return psf


class StormAnaCoefficient:
    """Wrapper class as an interface for MATLAB Spline calibration data."""
    def __init__(self, file, spline_obj=deepsmlm.generic.psf_kernel.CubicSplinePSF):
        """

        :param file: .mat file
        """
        self.calib_file = file
        np_container = np.load(self.calib_file)
        self.coeff = torch.from_numpy(np_container['coeff'])
        self.ref0 = (int((self.coeff.shape[0] - 1) / 2), int((self.coeff.shape[1] - 1) / 2), int((self.coeff.shape[2] - 1) / 2))
        self.zRange = tuple(np_container['zRange'])

        self.spline_obj = spline_obj

    def init_spline(self, xextent, yextent, zextent, img_shape):
        """
        Initialise Spline
        :return: spline instance
        """
        psf = self.spline_obj(xextent=xextent,
                              yextent=yextent,
                              zextent=self.zRange,
                              coeff=self.coeff,
                              ref0=self.ref0,
                              img_shape=img_shape)
        psf.print_basic_properties()
        return psf


if __name__ == '__main__':
    root = '/home/lucas/RemoteDeploymentTemp/deepsmlm/'
    csp_calib = 'data/Cubic Spline Coefficients/2019-02-19/000_3D_cal_640i_50_Z-stack_1_MMStack.ome_3dcal.mat'
    sp = SMAPSplineCoefficient(root + csp_calib)
    spline_psf = sp.init_spline((-0.5, 25.5), (-0.5, 25.5), img_shape=(26, 26))
    xxx = spline_psf.forward(torch.tensor([[15., 15., 0]]), torch.tensor([1.]))

    # storm_coefficient = '/Users/lucasmueller/Documents/Uni/EMBL/SMLM Challenge/Calibration/storm_ana_psf_coeff.npz'
    # sp = StormAnaCoefficient(storm_coefficient)
    # spline_psf_2 = sp.init_spline((-0.5, 25.5), (-0.5, 25.5), None, img_shape=(26, 26))
    # spline_psf_2.forward(torch.tensor([[15., 15., 0]]), torch.tensor([1.]))

    print('Done.')