from typing import Optional, Iterable, Callable

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


class MicroscopeMultiChannel:
    def __init__(
        self,
        psf: list[psf_kernel.PSF],
        noise: list[noise_lib.NoiseDistribution],
        frame_range: Optional[tuple[int, int]],
        ch_range: Optional[tuple[int, int]],
    ):
        """
        A microscope that has multi channels. Internally this is modelled as a list
        of individual microscopes.

        Args:
            psf: list of psf
            noise: list of noise
            frame_range: frame range among which frames are sampled
            ch_range: range of active channels
        """
        self._microscopes: list[Microscope] = [
            Microscope(psf=p, noise=n, frame_range=frame_range)
            for p, n in zip(psf, noise)
        ]
        self._ch_range = ch_range

    def forward(
        self,
        em: EmitterSet,
        bg: Optional[Iterable[torch.Tensor]] = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward emitters through multi channel microscope

        Args:
            em: emitters
            bg: list of bg with length equals to number of channels
            ix_low: lower frame index
            ix_high: upper frame index

        Returns:
            frames
        """

        em_by_channel = [em.icode[c] for c in range(*self._ch_range)]
        bg = [None] * len(em_by_channel) if bg is None else bg

        frames = [
            m.forward(e, bg=b, ix_low=ix_low, ix_high=ix_high)
            for m, e, b in zip(self._microscopes, em_by_channel, bg)
        ]
        frames = torch.stack(frames, dim=1)

        return frames


class MicroscopeChannelModifier:
    def __init__(self, ch_fn: list[Callable]):
        """
        Used to apply a transformation per channel on an EmitterSet.


        Args:
            ch_fn: list of callables taking and outputting an EmitterSet.
        """
        self._ch_fn = ch_fn

    def forward(self, em: EmitterSet) -> EmitterSet:
        em = [ch_fn(em) for ch_fn in self._ch_fn]
        em = EmitterSet.cat(em)
        return em


class EmitterCompositeAttributeModifier:
    def __init__(self, mod_fn: dict[str, Callable]):
        """
        Modify emitter attributes by independent callables.
        The order of the dictionary is the order in which the attributes are changed.

        Examples:
            `mod = EmitterCompositeAttributeModifier(
            {"xyz": lambda x: x/2, "phot": lambda p: p * 2}
            )`
            would divide the xyz coordinates by 2 and doubles the photon count.

        Args:
            mod_fn: dictionary of callables with key being the attribute to modify
        """
        self._mod_fn = mod_fn

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, em: EmitterSet) -> EmitterSet:
        em = em.clone()
        for attr, mod_fn in self._mod_fn.items():
            v = mod_fn(getattr(em, attr))
            setattr(em, attr, v)
        return em
