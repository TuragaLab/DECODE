"""
Here we provide some filtering on EmitterSets.
"""
from abc import ABC, abstractmethod
from deprecated import deprecated

import torch


class EmitterFilter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, em):
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


@deprecated(reason="I don't see use for this class anymore.")
class FrameFilter:
    """
    Simple class to filter out input frames when simulation may provide more
    """

    def __init__(self, n_frames):
        """

        Args:
            n_frames: (int) number of frames
        """
        self.n_frames = n_frames

    @staticmethod
    def parse(param):
        return FrameFilter(n_frames=param.HyperParameter.channels_in)

    def forward(self, x: torch.Tensor):
        """
        Assumes the frames to be of size N x C x H x W or C x H x W. The filtering is on the channel dimension.
        Args:
            x: input frames

        Returns:
            filtered input frames
        """
        dim = x.dim()

        if dim == 3:
            squeeze = True
            x_ = x.unsqueeze(0)
        else:
            squeeze = False

        ch = x_.size(1)

        assert (ch == self.n_frames or self.n_frames == 1)
        assert ch % 2 == 1

        if self.n_frames == 1:
            x_out = x_[:, [ch // 2]]
            return x_out.squeeze(0) if squeeze else x_out
        else:
            return x_.squeeze(0) if squeeze else x_
