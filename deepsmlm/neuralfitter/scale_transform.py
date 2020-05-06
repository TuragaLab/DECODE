import functools

import torch


class SpatialInterpolation:
    """
    Up or downscales by a given method.

    Attributes:
        dim (int): dimensionality for safety checks
    """

    def __init__(self, mode='nearest', size=None, scale_factor=None, impl=None):
        """

        Args:
            mode (string, None): mode which is used for interpolation. Those are the modes by the torch interpolation
            function
            impl (optional): override function for interpolation
        """

        if impl is not None:
            self._inter_impl = impl
        else:
            self._inter_impl = functools.partial(torch.nn.functional.interpolate,
                                                 mode=mode, size=size, scale_factor=scale_factor)

    @staticmethod
    def _unsq_call_sq(func, x: torch.Tensor, dim: int) -> any:
        """
        Unsqueeze input tensor until dimensionality 'dim' is matched and squeeze output before return

        Args:
            func:
            x:
            dim:

        Returns:

        """

        n_unsq = 0
        while x.dim() < dim:
            x.unsqueeze_(0)
            n_unsq += 1

        out = func(x)

        for _ in range(n_unsq):
            out.squeeze_(0)

        return out

    def forward(self, x: torch.Tensor):
        """
        Forward a tensor through the interpolation process.

        Args:
            x (torch.Tensor): arbitrary tensor complying with the interpolation function.
                Must have a batch and channel dimension.

        Returns:
            x_inter: interpolated tensor
        """

        return self._unsq_call_sq(self._inter_impl, x, 4)


class AmplitudeRescale:
    """
    Simple Processing that rescales the amplitude, i.e. the pixel values.

    Attributes:
        norm-value (float): Value to which to norm the data.
    """

    def __init__(self, norm_value: float):
        """

        Args:
            norm_value (float): reference value
        """
        self.norm_value = norm_value

    @staticmethod
    def parse(param):
        return AmplitudeRescale(norm_value=param.Scaling.in_count_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward the tensor and rescale it.

        Args:
            x (torch.Tensor):

        Returns:
            x_ (torch.Tensor): rescaled tensor

        """
        return x / self.norm_value


class OffsetRescale:
    """
       The purpose of this class is to rescale the (target) data from the network value world back to the real values.
       This class is used if we want to know the actual values and do not want to just use it for the loss.
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
    The purpose of this class is to scale the target data for the loss to an apropriate range.
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
