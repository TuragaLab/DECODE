import copy
from typing import Optional, Iterable, Callable

import torch

from . import noise as noise_lib
from . import psf_kernel
from ..emitter.emitter import EmitterSet
from ..generic import utils


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


class MicroscopeChannelSplitter:
    def __init__(self):
        pass


class MicroscopeChannelModifier:
    def __init__(self, ch_fn: list[Callable]):
        """
        Used to apply a transformation per channel on an EmitterSet.

        Warnings:
            - this treats channels as independent


        Args:
            ch_fn: list of callables taking and outputting an EmitterSet.
        """
        self._ch_fn = ch_fn

    def forward(self, em: EmitterSet) -> EmitterSet:
        em = [ch_fn(em) for ch_fn in self._ch_fn]
        em = EmitterSet.cat(em)
        return em


class EmitterCompositeAttributeModifier(utils.CompositeAttributeModifier):
    # lazy alias for modifying emitter attributes
    pass


class CoordTrafoMatrix:
    def __init__(self, m: torch.Tensor):
        """
        Transform coordinates by 3x3 matrix.

        Args:
            m: 3x3 matrix
        """
        self._m = m

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: coordinates of size `N x 3` (`N` being the batch dim).
        """
        return xyz @ self._m


class MultiChoricSplitter:
    def __init__(
        self,
        t: torch.Tensor,
        t_sig: Optional[torch.Tensor] = None,
    ):
        """
        Resembles a multi-choric beam splitter by a transmission matrix
        (which can be sampled).

        Args:
            t:
            t_sig:
        """
        self._t = t
        self._t_mu = copy.copy(t)
        self._t_sig = t_sig

    def forward(
        self, phot: torch.Tensor, color: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        if color is not None:
            phot = self._expand_col_by_index(phot, color, len(self._t))

        return phot @ self._t

    def sample_transmission_(self):
        # inplace

        self._t = self.sample_transmission()
        return self

    def sample_transmission(self):
        """
        Samples transmission matrix and renormalize it
        """
        return torch.normal(self._t_mu, self._t_sig)

    @staticmethod
    def _expand_col_by_index(x: torch.Tensor, ix: torch.LongTensor, ix_max: int):
        """
        Expands a one dim. tensor `x` to col dimension of size `ix_max` and puts the
        value at `ix`.

        Args:
            x: tensor to be expanded
            ix: col position
            ix_max: number of cols

        Examples:
            >>> _expand_col_by_index([1, 2], [1, 0], 3)
            [
                [0, 1, 0],
                [2, 0, 0]
            ]

        """
        x_out = x.unsqueeze(1).repeat(1, ix_max)

        # create bool with True where we should expand
        ix_bool = torch.zeros_like(x_out, dtype=torch.bool)
        ix_bool[torch.ones_like(x, dtype=torch.bool), ix] = True

        x_out *= ix_bool

        return x_out
