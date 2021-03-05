import functools
from typing import Tuple
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

    def __init__(self, scale: float = 1., offset: float = 0.):
        """

        Args:
            offset:
            scale (float): reference value
        """
        self.scale = scale if scale is not None else 1.
        self.offset = offset if offset is not None else 0.

    @staticmethod
    def parse(param):
        return AmplitudeRescale(scale=param.Scaling.input_scale,
                                offset=param.Scaling.input_offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward the tensor and rescale it.

        Args:
            x (torch.Tensor):

        Returns:
            x_ (torch.Tensor): rescaled tensor

        """
        return (x - self.offset) / self.scale


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

    @classmethod
    def parse(cls, param):
        return cls(scale_x=param.Scaling.dx_max,
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


class FourFoldInverseOffsetRescale(InverseOffsetRescale):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_no_bg = torch.cat([super(FourFoldInverseOffsetRescale, self).forward(x[..., i:(i + 5), :, :]) for i in range(0, 20, 5)], dim=1 if x.dim() == 4 else 0)
        bg = (x[..., [20], :, :] - self.mu_sig_bg[0]) / (self.mu_sig_bg[1] * self.buffer) ** self.power

        return torch.cat([x_no_bg, bg], 1 if x.dim() == 4 else 0)


class ParameterListRescale:

    def __init__(self, phot_max, z_max, bg_max):
        self.phot_max = phot_max
        self.z_max = z_max
        self.bg_max = bg_max

    def forward(self, x: torch.Tensor, mask: torch.Tensor, bg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if x.dim() not in (2, 3) or x.size(-1) != 4:
            raise ValueError(f"Unsupported shape of input {x.size()}")

        x = x.clone()
        x[..., 0] = x[..., 0] / self.phot_max
        x[..., 3] = x[..., 3] / self.z_max
        bg = bg / self.bg_max

        return x, mask, bg

    @classmethod
    def parse(cls, param):
        return cls(phot_max=param.Scaling.phot_max,
                   z_max=param.Scaling.z_max,
                   bg_max=param.Scaling.bg_max)


class InverseParamListRescale(ParameterListRescale):
    """
    Rescale network output trained with GMM Loss.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: model output

        Returns:
            torch.Tensor (rescaled model output)

        """

        if x.dim() != 4 or x.size(1) != 10:
            raise ValueError(f"Unsupported size of input {x.size()}")

        x = x.clone()

        x[:, 1] *= self.phot_max
        x[:, 5] *= self.phot_max  # sigma rescaling

        x[:, 4] *= self.z_max
        x[:, 8] *= self.z_max
        x[:, -1] *= self.bg_max

        return x
