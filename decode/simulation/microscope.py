import copy
from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable, Callable, Union, Sequence, Protocol

import torch
from deprecated import deprecated

from . import noise as noise_lib
from . import psf_kernel
from ..emitter.emitter import EmitterSet
from ..generic import utils


class XYZTransformation(Protocol):
    @abstractmethod
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ChoricTransformation(Protocol):
    @abstractmethod
    def forward(self, phot: torch.Tensor, color: torch.Tensor):
        raise NotImplementedError


class Microscope:
    def __init__(
        self,
        psf: psf_kernel.PSF,
        noise: Optional[noise_lib.NoiseDistribution] = None,
        frame_range: Optional[Union[int, tuple[int, int]]] = None,
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
        # default to 0 ... frame_range if int
        self._frame_range = frame_range if not isinstance(frame_range, int) else (0, frame_range)

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

        f = self._psf.forward(em.xyz_px, em.phot, em.frame_ix,
                              ix_low=ix_low, ix_high=ix_high)

        if bg is not None:
            f += bg
        if self._noise is not None:
            f = self._noise.forward(f)

        return f


class MicroscopeMultiChannel:
    def __init__(
        self,
        psf: list[psf_kernel.PSF],
        noise: list[Optional[noise_lib.NoiseDistribution]],
        frame_range: Optional[tuple[int, int]],
        ch_range: Optional[Union[int, tuple[int, int]]],
        trafo_xyz: Optional[XYZTransformation] = None,
        trafo_phot: Optional[ChoricTransformation] = None,
        stack: Optional[Union[str, Callable]] = None,
    ):
        """
        A microscope that has multi channels. Internally this is modelled as a list
        of individual microscopes.

        Args:
            psf: list of psf
            noise: list of noise
            frame_range: frame range among which frames are sampled
            ch_range: range of active channels
            trafo_xyz: channel-wise coordinate transformer
            trafo_phot: channel-wise photon transformer
            stack: stack function, None, `stack` or callable.
        """
        self._microscopes: list[Microscope] = [
            Microscope(psf=p, noise=n, frame_range=frame_range)
            for p, n in zip(psf, noise)
        ]
        self._ch_range = ch_range
        self._trafo_xyz = trafo_xyz
        self._trafo_phot = trafo_phot
        self._stack_impl = stack

    def _stack(self, x: Sequence[torch.Tensor]) -> Any:
        if self._stack_impl is None:
            return x
        if self._stack_impl == "stack":
            return torch.stack(x, dim=1)
        raise ValueError("Unsupported stack implementation.")

    def forward(
        self,
        em: EmitterSet,
        bg: Optional[Iterable[torch.Tensor]] = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> Any:
        """
        Forward emitters through multichannel microscope

        Args:
            em: emitters
            bg: list of bg with length equals to number of channels
            ix_low: lower frame index
            ix_high: upper frame index

        Returns:
            frames
        """
        if self._trafo_xyz is not None:
            em.xyz_px = self._trafo_xyz.forward(em.xyz_px)

        if self._trafo_phot is not None:
            em.phot = self._trafo_phot.forward(em.phot, em.code)

        if self._trafo_xyz is not None or self._trafo_phot is not None:
            em.code = em.infer_code()
            em = em.linearize()

        em_by_channel = [em.icode[c] for c in range(*self._ch_range)]
        bg = [None] * len(em_by_channel) if bg is None else bg

        frames = [
            m.forward(e, bg=b, ix_low=ix_low, ix_high=ix_high)
            for m, e, b in zip(self._microscopes, em_by_channel, bg)
        ]
        return self._stack(frames)


@deprecated(reason="Not necessary", version="0.11.1dev1")
class MicroscopeChannelSplitter:
    def __init__(self):
        raise NotImplementedError


@deprecated(reason="Not necessary", version="0.11.1dev1")
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


class XYZTransformationMatrix(XYZTransformation):
    def __init__(self, m: torch.Tensor):
        """
        Transform coordinates by (Nx)3x3 matrix. Outputs are coordinates, if the
        transformation matrix is batched, the different transformations per coordinate
        are treated as channel dimension, e.g. xyz of size 5 x 3 with matrix of size
        2 x 3 x 3 will lead to new xyz of size 5 x 2 x 3.

        Args:
            m: (Cx)3x3 matrix
        """
        if m.dim() not in {2, 3}:
            raise NotImplementedError("Not supported dimension of m.")
        self._m = m

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """

        Args:
            xyz: coordinates of size `N x 3` (`N` being the batch dim).
        """
        xyz = xyz @ self._m

        # batched transformations are treated as channels
        if self._m.dim() == 3:
            xyz = xyz.permute(1, 0, -1)

        return xyz


class MultiChoricSplitter(ChoricTransformation):
    def __init__(
        self,
        t: torch.Tensor,
        t_sig: Optional[torch.Tensor] = None,
        ix_low: Optional[int] = 0,
    ):
        """
        Resembles a multi-choric beam splitter by a transmission matrix
        (which can be sampled).

        Args:
            t: transmission matrix of size `C x C`
            t_sig: standard deviation of transmission matrix of size `C x C`
            ix_low: lower index of the channel range
        """
        self._t = t
        self._t_mu = copy.copy(t)
        self._t_sig = t_sig
        self._ix_low = ix_low

    def forward(
        self, phot: torch.Tensor, color: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        if color is not None:
            color = color - self._ix_low
            phot = self._expand_col_by_index(phot, color, len(self._t))

        return phot @ self._t

    def sample_transmission_(self):
        # inplace

        self._t = self.sample_transmission()
        return self

    def sample_transmission(self):
        """
        Samples transmission matrix and renormalizes it
        """
        t = torch.normal(self._t_mu, self._t_sig)
        # normalize twice to account for possible effect of clamp, otherwise
        # one can get nan
        t /= torch.sum(t, dim=1, keepdim=True)
        t = t.clamp(min=0.)
        t /= torch.sum(t, dim=1, keepdim=True)
        return t

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
