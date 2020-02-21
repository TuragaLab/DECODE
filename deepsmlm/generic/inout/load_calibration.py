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

        # ToDo: Update to new spline function
        raise NotImplementedError
        # psf = self.spline_obj(xextent=xextent,
        #                       yextent=yextent,
        #                       zextent=None,
        #                       img_shape=img_shape,
        #                       coeff=self.coeff,
        #                       ref0=self.ref0,
        #                       dz=self.dz)
        #
        # psf.print_basic_properties()
        # return psf
