from abc import ABC, abstractmethod  # abstract class
from collections import namedtuple

import numpy as np
import torch

from decode.simulation import psf_kernel as psf_kernel


class Background(ABC):
    """
    Abstract background class. All childs must implement a sample method.
    """

    _forward_modes = ('like', 'cum', 'tuple')
    _bg_return = namedtuple('bg_return', ['xbg', 'bg'])  # return arguments, x plus bg and bg term alone

    def __init__(self, forward_return: str = None):
        """

        Args:
            forward_return: determines the return of the forward function. 'like' returns a sample of the same size
                as the input, 'cum' adds the sample to the input and 'tuple' returns both the sum and the bg component
                alone.

        """
        super().__init__()

        self.forward_return = forward_return if forward_return is not None else 'tuple'

        self.sanity_check()

    def sanity_check(self):
        """
        Tests the sanity of the instance.
        """

        if self.forward_return not in self._forward_modes:
            raise ValueError(f"Forward return mode {self.forward_return} unsupported. "
                             f"Available modes are: {self._forward_modes}")

    @abstractmethod
    def sample(self, size: torch.Size, device=torch.device('cpu')) -> torch.Tensor:
        """
        Samples from background implementation in the specified size.

        Args:
            size: size of the sample
            device: where to put the data

        Returns:
            background sample

        """
        raise NotImplementedError

    def sample_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples background in the shape and on the device as the the input.

        Args:
            x: input

        Returns:
            background sample

        """
        return self.sample(size=x.size(), device=x.device)

    def forward(self, x: torch.Tensor):
        """
        Samples background in the same shape and on the same device as the input x.
        Depending on the 'forward_return' attribute the bg is
            - returned alone ('like')
            - added to the input ('cum')
            - is added and returned as tuple ('tuple')

        Args:
            x: input frames. Dimension :math:`(N,C,H,W)`

        Returns:
            (see above description)

        """

        bg = self.sample_like(x)

        if self.forward_return == 'like':
            return bg

        elif self.forward_return == 'cum':
            return bg + x

        elif self.forward_return == 'tuple':
            return self._bg_return(xbg=x + bg, bg=bg)

        else:
            raise ValueError


class UniformBackground(Background):
    """
    Spatially constant background (i.e. a constant offset).

    """

    def __init__(self, bg_uniform: (float, tuple) = None, bg_sampler=None, forward_return=None):
        """
        Adds spatially constant background.

        Args:
            bg_uniform (float or tuple of floats): background value or background range. If tuple (bg range) the value
                will be sampled from a random uniform.
            bg_sampler (function): a custom bg sampler function that can take a sample_shape argument

        """
        super().__init__(forward_return=forward_return)

        if (bg_uniform is not None) and (bg_sampler is not None):
            raise ValueError("You must either specify bg_uniform XOR a bg_distribution")

        if bg_sampler is None:
            if isinstance(bg_uniform, (list, tuple)):
                self._bg_distribution = torch.distributions.uniform.Uniform(*bg_uniform).sample
            else:
                self._bg_distribution = _get_delta_sampler(bg_uniform)

        else:
            self._bg_distribution = bg_sampler

    @staticmethod
    def parse(param):
        return UniformBackground(param.Simulation.bg_uniform)

    def sample(self, size, device=torch.device('cpu')):

        assert len(size) in (2, 3, 4), "Not implemented size spec."

        # create as many sample as there are batch-dims
        bg = self._bg_distribution(sample_shape=[size[0]] if len(size) >= 3 else torch.Size([]))

        # unsqueeze until we have enough dimensions
        if len(size) >= 3:
            bg = bg.view(-1, *((1,) * (len(size) - 1)))

        return bg.to(device) * torch.ones(size, device=device)


def _get_delta_sampler(val: float):
    def delta_sampler(sample_shape) -> float:
        return val * torch.ones(sample_shape)

    return delta_sampler


class BgPerEmitterFromBgFrame:
    """
    Extract a background value per localisation from a background frame. This is done by mean filtering.
    """

    def __init__(self, filter_size: int, xextent: tuple, yextent: tuple, img_shape: tuple):
        """

        Args:
            filter_size (int): size of the mean filter
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape (tuple): image shape
        """
        super().__init__()

        from decode.neuralfitter.utils import padding_calc as padcalc
        """Sanity checks"""
        if filter_size % 2 == 0:
            raise ValueError("ROI size must be odd.")

        self.filter_size = [filter_size, filter_size]
        self.img_shape = img_shape

        pad_x = padcalc.pad_same_calc(self.img_shape[0], self.filter_size[0], 1, 1)
        pad_y = padcalc.pad_same_calc(self.img_shape[1], self.filter_size[1], 1, 1)

        self.padding = torch.nn.ReplicationPad2d((pad_x, pad_x, pad_y, pad_y))  # to get the same output dim

        self.kernel = torch.ones((1, 1, filter_size, filter_size)) / (filter_size * filter_size)
        self.delta_psf = psf_kernel.DeltaPSF(xextent, yextent, img_shape)
        self.bin_x = self.delta_psf._bin_x
        self.bin_y = self.delta_psf._bin_y

    def _mean_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Actual magic

        Args:
            x: torch.Tensor of size N x C=1 x H x W

        Returns:
            (torch.Tensor) mean filter on frames
        """

        # put the kernel to the right device
        if x.size()[-2:] != torch.Size(self.img_shape):
            raise ValueError("Background does not match specified image size.")

        if self.filter_size[0] <= 1:
            return x

        self.kernel = self.kernel.to(x.device)
        x_mean = torch.nn.functional.conv2d(self.padding(x), self.kernel, stride=1, padding=0)  # since already padded
        return x_mean

    def forward(self, tar_em, tar_bg):

        if tar_bg.dim() == 3:
            tar_bg = tar_bg.unsqueeze(1)

        if len(tar_em) == 0:
            return tar_em

        local_mean = self._mean_filter(tar_bg)

        """Extract background values at the position where the emitter is and write it"""
        pos_x = tar_em.xyz[:, 0]
        pos_y = tar_em.xyz[:, 1]
        bg_frame_ix = (-int(tar_em.frame_ix.min()) + tar_em.frame_ix).long()

        ix_x = torch.from_numpy(np.digitize(pos_x.numpy(), self.bin_x, right=False) - 1)
        ix_y = torch.from_numpy(np.digitize(pos_y.numpy(), self.bin_y, right=False) - 1)

        """Kill everything that is outside"""
        in_frame = torch.ones_like(ix_x).bool()
        in_frame *= (ix_x >= 0) * (ix_x <= self.img_shape[0] - 1) * (ix_y >= 0) * (ix_y <= self.img_shape[1] - 1)

        tar_em.bg[in_frame] = local_mean[bg_frame_ix[in_frame], 0, ix_x[in_frame], ix_y[in_frame]]

        return tar_em
