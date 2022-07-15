from abc import ABC, abstractmethod
from typing import Any, TypeVar, Protocol, Optional

import torch

from . import process
from ..emitter import emitter
from ..simulation import microscope

# ToDo: move those to a utils place
T = TypeVar("T", bound="_Sliceable")


class _Sliceable(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self: T, i: slice) -> T:
        ...


class Sampler(ABC):
    @property
    @abstractmethod
    def frame(self) -> _Sliceable:
        raise NotImplementedError

    @property
    def emitter(self) -> emitter.EmitterSet:
        raise NotImplementedError

    @property
    def target(self) -> _Sliceable:
        raise NotImplementedError


class _SlicerDelayed:
    def __init__(self, fn, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._fn = fn

    def __getitem__(self, item) -> Any:
        args = [o[item] for o in self._args]
        kwargs = {k: v[item] for k, v in self._kwargs.items()}

        return self._fn(*args, **kwargs)


class SamplerSupervised(Sampler):
    def __init__(
        self,
        em: emitter.EmitterSet,
        bg: torch.Tensor,
        mic: microscope.Microscope,
        proc: process.Processing,
        delay: bool = True,
    ):
        super().__init__()

        self._em = em
        self._bg = bg
        self._mic = mic
        self._proc = proc
        self._delay = delay
        self._frames = None

    @property
    def emitter(self) -> emitter.EmitterSet:
        return self._em

    @property
    def frame(self) -> torch.Tensor:
        return self._frames

    @property
    def bg(self) -> torch.Tensor:
        return self._bg

    @property
    def input(self) -> _SlicerDelayed:
        return _SlicerDelayed(
            self._proc.input, frame=self.frame, em=self.emitter.iframe, bg=self.bg
        )

    @property
    def target(self) -> _SlicerDelayed:
        """
        Returns a delayed target, i.e. the compute graph is attached, the actual
        computation happens when the elements are accessed via slicing (`__getitem__`).
        """
        return _SlicerDelayed(self._proc.tar, em=self.emitter.iframe, bg=self.bg)

    def sample(self):
        self._frames = self._mic.forward(em=self._em, bg=self._bg)


class IxShifter:
    _pad_modes = (None, "same")

    def __init__(self, mode: str, window: int):
        self._mode = mode
        self._window = window

        if mode not in self._pad_modes:
            raise NotImplementedError

    def __call__(self, ix: int) -> int:
        if self._mode is None:
            # no padding means we need to shift indices, i.e. loose a few samples
            if ix < 0:
                raise ValueError("Negative indexing not supported.")
            ix = ix + (self._window - 1) // 2

        return ix


class IxWindow:
    def __init__(self, win: int, n: Optional[int]):
        """
        'Window' an index, that make convolution like.

        Args:
            win: window size
            n: data size

        Examples:
            >>> IxWindow(3)(0)
            [0, 0, 1]

            >>> IxWindow(3)(5)
            [4, 5, 6]
        """
        self._win = win
        self._n = n

    def __call__(self, ix: int) -> list[int]:
        hw = (self._win - 1) // 2  # half window without centre
        ix = torch.arange(ix - hw, ix + hw + 1).clamp(0)

        if self._n is not None:
            ix = ix.clamp(max=self._n - 1)

        return ix.tolist()
