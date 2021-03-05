import scipy.io as sio
import torch

import decode.simulation.psf_kernel as psf_kernel


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

    def init_spline(self, xextent, yextent, img_shape, device='cuda:0' if torch.cuda.is_available() else 'cpu', **kwargs):
        """
        Initializes the CubicSpline function

        Args:
            xextent:
            yextent:
            img_shape:
            device: on which device to simulate

        Returns:

        """
        psf = psf_kernel.CubicSplinePSF(xextent=xextent, yextent=yextent, img_shape=img_shape, ref0=self.ref0,
                                        coeff=self.coeff, vx_size=(1., 1., self.dz), device=device, **kwargs)

        return psf