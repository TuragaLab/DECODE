import numpy as np
import scipy.io as sio
import torch

import deepsmlm.generic.psf_kernel as psf_kernel


class SMAPSplineCoefficient:
    """Wrapper class as an interface for MATLAB Spline calibration data."""
    def __init__(self, calib_file):
        """
        Loads a calibration file from SMAP and the relevant meta information
        Args:
            file:
        """
        self.calib_file = calib_file
        self.calib_mat = sio.loadmat(self.calib_file, struct_as_record=False, squeeze_me=True)['SXY']

        self.coeff = torch.from_numpy(self.calib_mat.cspline.coeff)
        self.ref0 = (self.calib_mat.cspline.x0 - 1, self.calib_mat.cspline.x0 - 1, self.calib_mat.cspline.z0)
        self.dz = self.calib_mat.cspline.dz
        self.spline_roi_shape = self.coeff.shape[:3]

    def init_spline(self, xextent, yextent, img_shape, roi_size=None, cuda=torch.cuda.is_available()):
        """
        Initializes the CubicSpline function
        Args:
            xextent:
            yextent:
            img_shape:

        Returns:

        """
        psf = psf_kernel.CubicSplinePSF(xextent=xextent,
                                        yextent=yextent,
                                        img_shape=img_shape,
                                        roi_size=roi_size,
                                        coeff=self.coeff,
                                        vx_size=None,
                                        ref0=self.ref0,
                                        )

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
