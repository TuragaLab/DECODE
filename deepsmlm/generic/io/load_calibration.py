import scipy.io as sio
import torch

from deepsmlm.generic.psf_kernel import SplineCPP


class SMAPSplineCoefficient:
    """Wrapper class as an interface for MATLAB Spline calibration data."""
    def __init__(self, file):
        """

        :param file: .mat file
        """
        self.calib_file = file
        self.calib_mat = sio.loadmat(self.calib_file, struct_as_record=False, squeeze_me=True)['SXY']

        self.coeff = torch.from_numpy(self.calib_mat.cspline.coeff)
        self.ref0 = (self.calib_mat.cspline.x0, self.calib_mat.cspline.x0, self.calib_mat.cspline.z0)
        self.dz = self.calib_mat.cspline.dz
        self.spline_roi_shape = self.coeff.shape[:3]

    def init_spline(self, xextent, yextent, zextent, img_shape):
        """
        Initialise Spline
        :return: spline instance
        """
        return SplineCPP(xextent=xextent,
                         yextent=yextent,
                         zextent=zextent,
                         coeff=self.coeff,
                         ref0=self.ref0,
                         img_shape=img_shape)


if __name__ == '__main__':
    root = '/home/lucas/RemoteDeploymentTemp/deepsmlm/'
    csp_calib = 'data/Cubic Spline Coefficients/2019-02-19/000_3D_cal_640i_50_Z-stack_1_MMStack.ome_3dcal.mat'
    sp = SMAPSplineCoefficient(root + csp_calib)
    spline_psf = sp.init_spline((-0.5, 25.5), (-0.5, 25.5), None, img_shape=(26, 26))
