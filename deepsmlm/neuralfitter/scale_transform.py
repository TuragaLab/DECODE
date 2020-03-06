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

    def __init__(self, *, scale_x: float, scale_y: float, scale_z: float, scale_phot: float, mu_sig_bg=(None, None),
                 buffer=1., power=1.):
        """
        Assumes scale_x, scale_y, scale_z to be symmetric ranged, scale_phot, ranged between 0-1

        Args:
            scale_x (float): scale factor in x
            scale_y: scale factor in y
            scale_z: scale factor in z
            scale_phot: scale factor for photon values
            mu_sig_bg: offset and scaling for background
            buffer: buffer to extend the scales overall
            power: power factor
        """

        self.sc_x = scale_x
        self.sc_y = scale_y
        self.sc_z = scale_z
        self.sc_phot = scale_phot
        self.mu_sig_bg = mu_sig_bg
        self.buffer = buffer
        self.power = power

    @staticmethod
    def parse(param):
        return OffsetRescale(scale_x=param.Scaling.dx_max,
                             scale_y=param.Scaling.dy_max,
                             scale_z=param.Scaling.z_max,
                             scale_phot=param.Scaling.phot_max,
                             mu_sig_bg=param.Scaling.mu_sig_bg,
                             buffer=param.Scaling.linearisation_buffer)

    def return_inverse(self):
        """
        Returns the inverse counterpart of this class (instance).

        Returns:
            InverseOffSetRescale: Inverse counterpart.

        """
        return InverseOffsetRescale(scale_x=self.sc_x,
                                    scale_y=self.sc_y,
                                    scale_z=self.sc_z,
                                    scale_phot=self.sc_phot,
                                    mu_sig_bg=self.mu_sig_bg,
                                    buffer=self.buffer,
                                    power=self.power)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale the input (typically after the network).

        Args:
            x (torch.Tensor): input tensor N x 5/6 x H x W

        Returns:
            x_ (torch.Tensor): scaled

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

    def __init__(self, *, scale_x: float, scale_y: float, scale_z: float, scale_phot: float, mu_sig_bg=(None, None),
                 buffer=1., power=1.):
        """
        Assumes scale_x, scale_y, scale_z to be symmetric ranged, scale_phot, ranged between 0-1

        Args:
            scale_x (float): scale factor in x
            scale_y: scale factor in y
            scale_z: scale factor in z
            scale_phot: scale factor for photon values
            mu_sig_bg: offset and scaling for background
            buffer: buffer to extend the scales overall
            power: power factor
        """
        super().__init__(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z, scale_phot=scale_phot,
                         mu_sig_bg=mu_sig_bg, buffer=buffer, power=power)

    @staticmethod
    def parse(param):
        return InverseOffsetRescale(scale_x=param.Scaling.dx_max,
                                    scale_y=param.Scaling.dy_max,
                                    scale_z=param.Scaling.z_max,
                                    scale_phot=param.Scaling.phot_max,
                                    mu_sig_bg=param.Scaling.mu_sig_bg,
                                    buffer=param.Scaling.linearisation_buffer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse scale transformation (typically before the network).

        Args:
            x (torch.Tensor): input tensor N x 5/6 x H x W

        Returns:
            x_ (torch.Tensor): (inverse) scaled

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
