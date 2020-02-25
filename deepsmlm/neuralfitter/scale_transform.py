import functools
import torch


class SpatialInterpolation:
    """
    Up or downscales by a given method.

    Attributes:
        dim (int): dimensionality for safety checks
    """
    def __init__(self, dim=2, mode='nearest', size=None, scale_factor=None, impl=None):
        """

        Args:
            mode (string): mode which is used for interpolation. Those are the modes by the torch interpolation function
            impl (optional): override function for interpolation
        """
        self.dim = dim

        if impl is not None:
            self._inter_impl = impl
        else:
            self._inter_impl = functools.partial(torch.nn.functional.interpolate,
                                                 mode=mode, size=size, scale_factor=scale_factor)

    def forward(self, x: torch.Tensor):
        """
        Forward a tensor through the interpolation process.

        Args:
            x (torch.Tensor): arbitrary tensor complying with the interpolation function.
                Must have a batch and channel dimension.

        Returns:
            x_inter: interpolated tensor
        """
        if x.dim() != self.dim + 2:  # dimensionality plus channel and batch
            raise ValueError(f"Dimensionality does not comply to specified initialisation. Expected tensor with "
                             "minibatch and channels plus {self.dim} dimensions (height, width ...)")

        return self._inter_impl(x)


class AmplitudeRescale:
    """
    Simple Processing that rescales the amplitude, i.e. the pixel values
    """
    def __init__(self, max_frame_count: float):
        self.max_frame_count = max_frame_count

    @staticmethod
    def parse(param):
        return AmplitudeRescale(max_frame_count=param['Scaling']['in_count_max'])

    def forward(self, frames):
        return frames / self.max_frame_count


class OffsetRescale:
    """
       The purpose of this class is to rescale the data from the network value world back to useful values.
       This class is used after the network output.
       """
    def __init__(self, scale_x: float, scale_y: float, scale_z: float, scale_phot: float, mu_sig_bg, buffer=1., power=1.):
        """
        Assumes scale_x, scale_y, scale_z to be symmetric ranged, scale_phot, ranged between 0-1
        :param scale_x:
        :param scale_y:
        :param scale_z:
        :param scale_phot:
        :param scale_bg:
        :param buffer: to extend the original range a little bit, to use the more linear parts of a sigmoidal fct.
        :param px_size: scale to nm.
        Does not apply to probability channel 0.
        """

        self.sc_x = scale_x
        self.sc_y = scale_y
        self.sc_z = scale_z
        self.sc_phot = scale_phot
        self.mu_sig_bg = mu_sig_bg
        self.buffer = buffer
        self.power = power

    @staticmethod
    def parse(param: dict):
        """

        :param param: param dictionary
        :return:
        """
        return OffsetRescale(param['Scaling']['dx_max'],
                      param['Scaling']['dy_max'],
                      param['Scaling']['z_max'],
                      param['Scaling']['phot_max'],
                      param['Scaling']['mu_sig_bg'],
                      param['Scaling']['linearisation_buffer'])

    def forward(self, x):
        """
        Scale the NN output to the apropriate scale
        :param x: (torch.tensor, N x 5 x H x W) or 5 x H x W
        :return:
        """
        if x.dim() == 3:
            x.unsqueeze_(0)
            squeeze_before_return = True
        else:
            squeeze_before_return = False

        x_ = x.clone()

        x_[:, 1, :, :] *= (self.sc_phot * self.buffer) ** self.power
        x_[:, 2, :, :] *= (self.sc_x * self.buffer) ** self.power
        x_[:, 3, :, :] *= (self.sc_y * self.buffer) ** self.power
        x_[:, 4, :, :] *= (self.sc_z * self.buffer) ** self.power
        if x_.size(1) == 6:
            x_[:, 5, :, :] *= (self.mu_sig_bg[1] * self.buffer) ** self.power
            x_[:, 5, :, :] += self.mu_sig_bg[0]

        if squeeze_before_return:
            return x_.squeeze(0)
        else:
            return x_


class InverseOffsetRescale(OffsetRescale):
    """
    The purpose of this class is to provide the output to the network, i.e. scaling the data between -1,1 or 0,1.
    This class is used before the network.
    """
    def __init__(self, scale_x: float, scale_y: float, scale_z: float, scale_phot: float, mu_sig_bg, buffer=1., power=1.):
        """
        Assumes scale_x, scale_y, scale_z to be symmetric ranged, scale_phot, ranged between 0-1
        :param scale_x:
        :param scale_y:
        :param scale_z:
        :param scale_phot:
        """
        super().__init__(scale_x, scale_y, scale_z, scale_phot, mu_sig_bg, buffer, power)

    @staticmethod
    def parse(param):
        """

        :param param: parameter dictionary
        :return: instance
        """
        return InverseOffsetRescale(param['Scaling']['dx_max'],
                                    param['Scaling']['dy_max'],
                                    param['Scaling']['z_max'],
                                    param['Scaling']['phot_max'],
                                    param['Scaling']['mu_sig_bg'],
                                    param['Scaling']['linearisation_buffer'])

    def forward(self, x):
        """
        Scale the original output to the NN range
        :param x:
        :return:
        """
        if x.dim() == 3:
            x.unsqueeze_(0)
            squeeze_before_return = True
        else:
            squeeze_before_return = False

        x_ = x.clone()

        x_[:, 1, :, :] /= (self.sc_phot * self.buffer) ** self.power
        x_[:, 2, :, :] /= (self.sc_x * self.buffer) ** self.power
        x_[:, 3, :, :] /= (self.sc_y * self.buffer) ** self.power
        x_[:, 4, :, :] /= (self.sc_z * self.buffer) ** self.power
        if x_.size(1) == 6:
            x_[:, 5, :, :] -= self.mu_sig_bg[0]
            x_[:, 5, :, :] /= (self.mu_sig_bg[1] * self.buffer) ** self.power

        if squeeze_before_return:
            return x_.squeeze(0)
        else:
            return x_
