from abc import ABC, abstractmethod  # abstract class
from typing import Callable, Optional, Union

import numpy as np
import torch

from decode.simulation import psf_kernel as psf_kernel


class Background(ABC):
    def __init__(self, size: Union[tuple[int, ...], torch.Size], device: str = "cpu"):
        """
        Background
        """
        super().__init__()

        self._size = size
        self._device = device

    @abstractmethod
    def sample(
        self, size: Union[tuple[int, ...], torch.Size], device: str = "cpu"
    ) -> torch.Tensor:
        """
        Samples from background implementation in the specified size.

        Args:
            size: size of the sample
            device: from which device to sample from

        """
        raise NotImplementedError

    def sample_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples background in the shape and on the device as the input.

        Args:
            x: input

        Returns:
            background sample

        """
        return self.sample(size=x.size(), device=x.device)

    def _arg_defaults(self, size: Optional[Union[tuple[int, ...], torch.Size]], device: str):
        """Overwrite optional args with instance defaults."""
        size = self._size if size is None else size
        device = self._device if device is None else device

        return size, device


class BackgroundUniform(Background):
    def __init__(
        self,
        bg: Union[float, tuple, Callable],
        size: Optional[Union[tuple[int, ...], torch.Size]] = None,
        device: str = "cpu",
    ):
        """
        Spatially constant background (i.e. a constant offset).

        Args:
            bg: background value, range or callable to sample from
            size:
            device:

        """
        super().__init__(size=size, device=device)

        if callable(bg):
            self._bg_dist = bg
        if isinstance(bg, (list, tuple)):
            self._bg_dist = torch.distributions.uniform.Uniform(*bg).sample
        else:
            self._bg_dist = _get_delta_sampler(bg)

    def sample(
        self,
        size: Optional[Union[tuple[int, ...], torch.Size]] = None,
        device: str = "cpu",
    ):
        size = size if size is not None else self._size

        if len(size) not in (2, 3, 4):
            raise NotImplementedError("Not implemented size spec.")

        # create as many sample as there are batch-dims
        bg = self._bg_dist(
            sample_shape=[size[0]] if len(size) >= 3 else torch.Size([])
        )

        # unsqueeze until we have enough dimensions
        if len(size) >= 3:
            bg = bg.view(-1, *((1,) * (len(size) - 1)))

        return bg.to(device) * torch.ones(size, device=device)


def _get_delta_sampler(val: float):
    def delta_sampler(sample_shape) -> float:
        return val * torch.ones(sample_shape)

    return delta_sampler


# ToDo: It is not immediately apparent in the doc strings what this thing really does
class BgPerEmitterFromBgFrame:
    def __init__(
        self, filter_size: int, xextent: tuple, yextent: tuple, img_shape: tuple
    ):
        """
        Extract a background value per localisation from a background frame.
        This is done by mean filtering.

        Args:
            filter_size (int): size of the mean filter
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape (tuple): image shape
        """
        super().__init__()

        from decode.neuralfitter.utils import padding_calc as padcalc

        if filter_size % 2 == 0:
            raise ValueError("ROI size must be odd.")

        self.filter_size = [filter_size, filter_size]
        self.img_shape = img_shape

        pad_x = padcalc.pad_same_calc(self.img_shape[0], self.filter_size[0], 1, 1)
        pad_y = padcalc.pad_same_calc(self.img_shape[1], self.filter_size[1], 1, 1)

        # to get the same output dim
        self.padding = torch.nn.ReplicationPad2d((pad_x, pad_x, pad_y, pad_y))

        self.kernel = torch.ones((1, 1, filter_size, filter_size)) / (
            filter_size * filter_size
        )
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
        x_mean = torch.nn.functional.conv2d(
            self.padding(x), self.kernel, stride=1, padding=0
        )  # since already padded
        return x_mean

    def forward(self, tar_em, tar_bg):

        if tar_bg.dim() == 3:
            tar_bg = tar_bg.unsqueeze(1)

        if len(tar_em) == 0:
            return tar_em

        local_mean = self._mean_filter(tar_bg)

        # extract background values at the position where the emitter is and write it
        pos_x = tar_em.xyz[:, 0]
        pos_y = tar_em.xyz[:, 1]
        bg_frame_ix = (-int(tar_em.frame_ix.min()) + tar_em.frame_ix).long()

        ix_x = torch.from_numpy(np.digitize(pos_x.numpy(), self.bin_x, right=False) - 1)
        ix_y = torch.from_numpy(np.digitize(pos_y.numpy(), self.bin_y, right=False) - 1)

        # kill everything that is outside
        in_frame = torch.ones_like(ix_x).bool()
        in_frame *= (
            (ix_x >= 0)
            * (ix_x <= self.img_shape[0] - 1)
            * (ix_y >= 0)
            * (ix_y <= self.img_shape[1] - 1)
        )

        tar_em.bg[in_frame] = local_mean[
            bg_frame_ix[in_frame], 0, ix_x[in_frame], ix_y[in_frame]
        ]

        return tar_em
