from abc import ABC, abstractmethod  # abstract class
from typing import Optional, Union

import torch
from deprecated import deprecated

from ...emitter import emitter
from . import to_emitter
from .. import scale_transform
from .. import coord_transform


class PostProcessing(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> emitter.EmitterSet:
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


class PostProcessingGaussianMixture(PostProcessing):
    def __init__(
        self,
        scaler: Optional[scale_transform.ScalerModelOutput] = None,
        coord_convert: Optional[coord_transform.Offset2Coordinate] = None,
        frame_to_emitter: Optional[to_emitter.ToEmitter] = None,
        coord_ch_ix: tuple[int, ...] = (2, 3),
    ):
        """

        Args:
            scaler: re-scales model output
            coord_convert: convert coordinates
            frame_to_emitter: extracts emitters from frame
            coord_ch_ix: define which channels are x, y to be passed on to the coord
             converter
        """
        super().__init__()
        self._scaler = scaler
        self._coord = coord_convert
        self._frame2em = frame_to_emitter
        self._coord_ch_ix = coord_ch_ix

    def forward(self, x: torch.Tensor) -> Union[emitter.EmitterSet, torch.Tensor]:
        """
        Applies post-processing pipeline

        Args:
            x:

        Returns:

        """
        if self._scaler is not None:
            x = self._scaler.forward(x)
        if self._coord is not None:
            x[..., self._coord_ch_ix, :, :] = self._coord.forward(
                x[..., self._coord_ch_ix, :, :]
            )
        if self._frame2em is not None:
            x = self._frame2em.forward(x)

        return x


class PostProcessingLookUp(to_emitter.ToEmitterLookUpPixelwise, PostProcessing):
    # quasi-alias for backwards compatibility
    pass


class PostProcessingSpatialIntegration(to_emitter.ToEmitterLookUpPixelwise):
    # alias
    pass


@deprecated(version="0.11", reason="Deprecated in favour `ToEmitterEmpty`")
class NoPostProcessing(to_emitter.ToEmitterEmpty, PostProcessing):
    pass


@deprecated(version="0.11", reason="Not in use or developed anymore.")
class PostProcessingConsistency(PostProcessing):
    """
    PostProcessing implementation that divides the output in hard and easy samples.
    Easy samples are predictions in which we have a single one hot pixel in the
    detection channel, hard samples are pixels in the detection channel where the
    adjacent pixels are also active (i.e. above a certain initial threshold).
    """

    def forward(self, x: torch.Tensor) -> emitter.EmitterSet:
        raise NotImplementedError
