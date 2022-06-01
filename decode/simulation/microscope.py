from typing import Optional

import torch

from . import noise as noise_lib
from . import psf_kernel
from ..emitter.emitter import EmitterSet


class Microscope:
    def __init__(
        self,
        psf: psf_kernel.PSF,
        noise: noise_lib.NoiseDistribution,
        frame_range: Optional[tuple[int, int]] = None,
    ):
        """
        Microscope consisting of psf and noise model.

        Args:
            psf: point spread function
            noise: noise model
            frame_range: frame range in which to sample
        """
        self._psf = psf
        self._noise = noise
        self._frame_range = frame_range

    def forward(
        self,
        em: EmitterSet,
        bg: Optional[torch.Tensor] = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward emitter and background through microscope and return frames.

        Args:
            em: emitters
            bg: background
            ix_low: lower frame index
            ix_high: upper frame index

        """
        ix_low = ix_low if ix_low is not None else self._frame_range[0]
        ix_high = ix_high if ix_high is not None else self._frame_range[1]

        f = self._psf.forward(em.xyz_px, em.phot, em.frame_ix, ix_low, ix_high)
        if bg is not None:
            f += bg
        f = self._noise.forward(f)

        return f
