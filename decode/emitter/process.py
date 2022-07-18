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


class EmitterIdentity(EmitterProcess):
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


@deprecated(reason="Use generic filter.", version="0.11.0")
class TarFrameEmitterFilter(EmitterProcess):
    """Filters the emitters on the target frame index."""

    def __init__(self, tar_ix=0):
        """

        Args:
            tar_ix: (int) index of the target frame
        """
        super().__init__()
        self.tar_ix = tar_ix

    def forward(self, em: EmitterSet) -> EmitterSet:
        """

        Args:
            em: (EmitterSet)

        Returns:
            em: (EmitterSet) filtered set of emitters
        """
        ix = em.frame_ix == self.tar_ix
        return em[ix]


@deprecated(reason="Use generic filter.", version="0.11.0")
class PhotonFilter(EmitterProcess):

    def __init__(self, th):
        """

        Args:
            th: (int, float) photon threshold
        """
        super().__init__()
        self.th = th

    def forward(self, em: EmitterSet) -> EmitterSet:
        """

        Args:
            em: (EmitterSet)

        Returns:
            em: (EmitterSet) filtered set of emitters
        """
        ix = em.phot >= self.th
        return em[ix]


class RemoveOutOfFOV(EmitterProcess):
    def __init__(self, xextent, yextent, zextent=None, xy_unit=None):
        """
        Processing class to remove emitters that are outside a specified extent.
        The lower / left respective extent limits are included, the right / upper extent limit is excluded / open.

        Args:
            xextent: extent of allowed field in x direction
            yextent: extent of allowed field in y direction
            zextent: (optional) extent of allowed field in z direction
            xy_unit: which xy is considered
        """
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.xy_unit = xy_unit

    def clean_emitter(self, xyz):
        """
        Returns index of emitters that are inside the specified extent.

        Args:
            xyz:

        Returns:

        """

        is_emit = (xyz[:, 0] >= self.xextent[0]) * (xyz[:, 0] < self.xextent[1]) * \
                  (xyz[:, 1] >= self.yextent[0]) * (xyz[:, 1] < self.yextent[1])

        if self.zextent is not None:
            is_emit *= (xyz[:, 2] >= self.zextent[0]) * (xyz[:, 2] < self.zextent[1])

        return is_emit

    def forward(self, em: EmitterSet) -> EmitterSet:
        """Removes emitters that are outside of the specified extent."""

        if self.xy_unit is None:
            em_mat = em.xyz
        elif self.xy_unit == 'px':
            em_mat = em.xyz_px
        elif self.xy_unit == 'nm':
            em_mat = em.xyz_nm
        else:
            raise ValueError(f"Unsupported xy unit: {self.xy_unit}")

        is_emit = self.clean_emitter(em_mat)

        return em[is_emit]
