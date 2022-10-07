from abc import ABC, abstractmethod  # abstract class

import torch
from deprecated import deprecated

from ...emitter.emitter import EmitterSet
from . import to_emitter


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


class LookUpPostProcessing(to_emitter.ToEmitterLookUpPixelwise, PostProcessing):
    # quasi-alias for backwards compatibility
    pass


class SpatialIntegration(to_emitter.ToEmitterLookUpPixelwise):
    # alias
    pass


@deprecated(version="0.11", reason="Deprecated in favour `ToEmitterEmpty`")
class NoPostProcessing(to_emitter.ToEmitterEmpty, PostProcessing):
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


