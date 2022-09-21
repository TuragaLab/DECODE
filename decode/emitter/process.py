from abc import ABC, abstractmethod
from deprecated import deprecated

import torch

from .emitter import EmitterSet


class EmitterProcess(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, em: EmitterSet) -> EmitterSet:
        """
        Forwards a set of emitters through the filter implementation

        Args:
            em: emitters

        """
        return em


class EmitterProcessNoOp(EmitterProcess):
    def forward(self, em: EmitterSet) -> EmitterSet:
        return em


class EmitterFilterGeneric(EmitterProcess):
    def __init__(self, **kwargs):
        """
        Generic emitter filter.

        Args:
            **kwargs: use emitter attribute and function that returns boolean

        Examples:
            # filters out emitters with less than 100 photons
             >>> f = EmitterFilterGeneric(phot=lambda p: p >= 100)
        """
        super().__init__()

        self._attr_fn = kwargs

    def forward(self, em: EmitterSet) -> EmitterSet:
        is_okay = torch.ones(len(em), dtype=torch.bool)

        for k, fn in self._attr_fn.items():
            is_okay *= fn(getattr(em, k))

        return em[is_okay]


class EmitterFilterFoV(EmitterProcess):
    def __init__(
        self,
        xextent: tuple[float, float],
        yextent: tuple[float, float],
        zextent=None,
        xy_unit="px",
    ):
        """
        Removes emitters that are outside a specified extent.
        The lower / left respective extent limits are included,
        the right / upper extent limit is excluded / open.

        Args:
            xextent: extent of allowed field in x direction
            yextent: extent of allowed field in y direction
            zextent: (optional) extent of allowed field in z direction
            xy_unit:
        """
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.xy_unit = xy_unit

    def _clean_emitter(self, xyz) -> torch.BoolTensor:
        """
        Returns index of emitters that are inside the specified extent.

        Args:
            xyz:

        Returns:

        """

        is_emit = (
            (xyz[:, 0] >= self.xextent[0])
            * (xyz[:, 0] < self.xextent[1])
            * (xyz[:, 1] >= self.yextent[0])
            * (xyz[:, 1] < self.yextent[1])
        )

        if self.zextent is not None:
            is_emit *= (xyz[:, 2] >= self.zextent[0]) * (xyz[:, 2] < self.zextent[1])

        return is_emit

    def forward(self, em: EmitterSet) -> EmitterSet:
        """Removes emitters that are outside of the specified extent."""

        if self.xy_unit is None:
            em_mat = em.xyz
        elif self.xy_unit == "px":
            em_mat = em.xyz_px
        elif self.xy_unit == "nm":
            em_mat = em.xyz_nm
        else:
            raise ValueError(f"Unsupported xy unit: {self.xy_unit}")

        is_emit = self._clean_emitter(em_mat)

        return em[is_emit]


class EmitterFilterFrame(EmitterProcess):
    def __init__(self, ix_low: int, ix_high: int, shift: int):
        """
        Filter emitters by frame. Thin wrapper around `em.get_subset_frame`.

        Args:
            ix_low: lower frame ix
            ix_high: upper frame ix
            shift: shift frames by
        """
        super().__init__()
        self._ix_low = ix_low
        self._ix_high = ix_high
        self._shift = shift

    def forward(self, em: EmitterSet) -> EmitterSet:
        return em.get_subset_frame(self._ix_low, self._ix_high, self._shift)


@deprecated(reason="Use generic filter.", version="0.11.0")
class TarFrameEmitterFilter(EmitterProcess):
    pass


@deprecated(reason="Use generic filter.", version="0.11.0")
class PhotonFilter(EmitterProcess):
    pass
