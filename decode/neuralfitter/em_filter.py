"""
Here we provide some filtering on EmitterSets.
"""
from abc import ABC, abstractmethod
from ..generic import EmitterSet


class EmitterFilter(ABC):
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


class NoEmitterFilter(EmitterFilter):
    """The no filter"""

    def forward(self, em):
        return em


class TarEmitterFilter(EmitterFilter):
    """Filters the emitters on the target frame index."""

    def __init__(self, tar_ix=0):
        """

        Args:
            tar_ix: (int) index of the target frame
        """
        super().__init__()
        self.tar_ix = tar_ix

    def forward(self, em):
        """

        Args:
            em: (EmitterSet)

        Returns:
            em: (EmitterSet) filtered set of emitters
        """
        ix = em.frame_ix == self.tar_ix
        return em[ix]


class PhotonFilter(EmitterFilter):
    """Filter on the photon count."""

    def __init__(self, th):
        """

        Args:
            th: (int, float) photon threshold
        """
        super().__init__()
        self.th = th

    def forward(self, em):
        """

        Args:
            em: (EmitterSet)

        Returns:
            em: (EmitterSet) filtered set of emitters
        """
        ix = em.phot >= self.th
        return em[ix]
