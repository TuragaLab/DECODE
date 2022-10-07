import warnings
from abc import ABC, abstractmethod  # abstract class
from typing import Any, Union, Callable

import numpy as np
import scipy
import torch
from deprecated import deprecated
from sklearn.cluster import AgglomerativeClustering

from decode.evaluation import match_emittersets
from decode.neuralfitter.utils.probability import binom_pdiverse
from ..emitter import emitter
from ..emitter.emitter import EmitterSet, EmptyEmitterSet
from ..generic import utils
from .processing import to_emitter


class PostProcessing(ABC):
    _return_types = ("batch-set", "frame-set")

    def __init__(self, xy_unit, px_size):
        """

        Args:
            one instance of EmitterSet will be returned per forward call, if 'frame-set' a tuple of EmitterSet one
            per frame will be returned
            sanity_check (bool): perform sanity check
        """

        super().__init__()
        self.xy_unit = xy_unit
        self.px_size = px_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> EmitterSet:
        """
        Forward anything through the post-processing and return an EmitterSet

        Args:
            x:

        Returns:
            EmitterSet or list: Returns as EmitterSet or as list of EmitterSets

        """
        raise NotImplementedError

    def skip_if(self, x) -> bool:
        """
        Skip post-processing when a certain condition is met and implementation would fail, i.e. to many
        bright pixels in the detection channel. Default implementation returns False always.

        Args:
            x: network output

        Returns:
            bool: returns true when post-processing should be skipped
        """
        return False


@deprecated(version="0.11", reason="Deprecated in favour `ToEmitterEmpty`")
class NoPostProcessing(to_emitter.ToEmitterEmpty, PostProcessing):
    pass


class LookUpPostProcessing(to_emitter.ToEmitterLookUpPixelwise, PostProcessing):
    # quasi-alias for backwards compatibility
    pass


class SpatialIntegration(to_emitter.ToEmitterLookUpPixelwise):
    # alias
    pass


@deprecated(version="0.11", reason="Not in use or developed anymore.")
class ConsistencyPostprocessing(PostProcessing):
    """
    PostProcessing implementation that divides the output in hard and easy samples.
    Easy samples are predictions in which we have a single one hot pixel in the
    detection channel, hard samples are pixels in the detection channel where the
    adjacent pixels are also active (i.e. above a certain initial threshold).
    """
    def forward(self, x: torch.Tensor) -> EmitterSet:
        raise NotImplementedError


class EmitterBackgroundByFrame:
    def __init__(
            self, filter_size: int, xextent: tuple, yextent: tuple, img_shape: tuple
    ):
        """
        Extract a background value per localisation from a background frame.
        What this does is performing mean filter smoothing on the background frame
        and then picking up the value at the pixel where the emitter is located.

        Args:
            filter_size (int): size of the mean filter
            xextent (tuple): extent in x
            yextent (tuple): extent in y
            img_shape (tuple): image shape
        """
        # Todo: Refactor filter as sepearte class and inject it
        from decode.neuralfitter.utils import padding_calc as padcalc

        if filter_size % 2 == 0:
            raise ValueError("ROI size must be odd.")

        self._filter_size = [filter_size, filter_size]
        self._img_shape = img_shape

        pad_x = padcalc.pad_same_calc(self._img_shape[0], self._filter_size[0], 1, 1)
        pad_y = padcalc.pad_same_calc(self._img_shape[1], self._filter_size[1], 1, 1)

        # to get the same output dim
        self._padding = torch.nn.ReplicationPad2d((pad_x, pad_x, pad_y, pad_y))

        self._kernel = torch.ones((1, 1, filter_size, filter_size)) / (
                filter_size * filter_size
        )
        self._bin_x, self._bin_y, *_ = utils.frame_grid(img_shape, xextent, yextent)

    def _mean_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean filter

        Args:
            x: torch.Tensor of size N x C=1 x H x W

        Returns:
            (torch.Tensor) mean filter on frames
        """

        # put the kernel to the right device
        if x.size()[-2:] != torch.Size(self._img_shape):
            raise ValueError("Background does not match specified image size.")

        if self._filter_size[0] <= 1:
            return x

        self._kernel = self._kernel.to(x.device)
        x_mean = torch.nn.functional.conv2d(
            self._padding(x), self._kernel, stride=1, padding=0
        )  # since already padded
        return x_mean

    def forward(self, em: emitter.EmitterSet, bg: torch.Tensor) -> emitter.EmitterSet:
        """

        Args:
            em: emitter to fill out background
            bg: background tensor

        Returns:

        """

        if bg.dim() == 3:
            bg = bg.unsqueeze(1)

        if len(em) == 0:
            return em

        local_mean = self._mean_filter(bg)

        # extract background values at the position where the emitter is and write it
        pos_x = em.xyz[:, 0]
        pos_y = em.xyz[:, 1]
        bg_frame_ix = (-int(em.frame_ix.min()) + em.frame_ix).long()

        ix_x = torch.from_numpy(
            np.digitize(pos_x.numpy(), self._bin_x, right=False) - 1
        )
        ix_y = torch.from_numpy(
            np.digitize(pos_y.numpy(), self._bin_y, right=False) - 1
        )

        # kill everything that is outside
        in_frame = torch.ones_like(ix_x).bool()
        in_frame *= (
                (ix_x >= 0)
                * (ix_x <= self._img_shape[0] - 1)
                * (ix_y >= 0)
                * (ix_y <= self._img_shape[1] - 1)
        )

        if em.bg is None:
            em.bg = float("nan") * torch.ones_like(em.xyz[..., 0])

        em.bg[in_frame] = local_mean[
            bg_frame_ix[in_frame], 0, ix_x[in_frame], ix_y[in_frame]
        ]
        return em
