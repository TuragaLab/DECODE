from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Protocol, Optional

import torch

from . import process
from .utils import indexing
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
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

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


T = TypeVar("T")


class _SlicerDelayed:
    def __init__(self, fn: Callable[..., T], *args, **kwargs):
        """
        Returns a sliceable handle and executes a function on __getitem__ where input
        arguments are then sliced and passed on to the function. Useful for delaying
        function executions.

        Args:
            fn:
            *args:
            **kwargs:
        """
        self._args = args
        self._kwargs = kwargs
        self._fn = fn

    def __getitem__(self, item) -> T:
        args = [o[item] for o in self._args]
        kwargs = {k: v[item] for k, v in self._kwargs.items()}

        return self._fn(*args, **kwargs)


class SamplerSupervised(Sampler):
    def __init__(
        self,
        em: emitter.EmitterSet,
        bg: Optional[torch.Tensor],
        mic: microscope.Microscope,
        proc: process.Processing,
        window: Optional[int] = 1,
        bg_mode: Optional[str] = None,
    ):
        """

        Args:
            em:
            bg:
            mic:
            proc:
            window:
            bg_mode: `global` or `sample`
        """
        super().__init__()

        self._em = em
        self._bg = bg
        self._mic = mic
        self._proc = proc
        self._window = window
        self._bg_mode = bg_mode

        self._frame = None
        self._frame_samples = None  # must be set toegther with _frame

    @property
    def emitter(self) -> emitter.EmitterSet:
        return self._em

    @property
    def frame(self) -> torch.Tensor:
        return self._frame

    @frame.setter
    def frame(self, v: torch.Tensor):
        self._frame = v
        self._frame_samples = indexing.IxWindow(self._window, None).attach(self._frame)

    @property
    def frame_samples(self) -> _Sliceable:
        return self._frame_samples

    @property
    def bg(self) -> torch.Tensor:
        return self._bg

    @property
    def input(self) -> _SlicerDelayed:
        return _SlicerDelayed(
            self._proc.input, frame=self.frame_samples, em=self.emitter.iframe, aux=self.bg
        )

    @property
    def target(self) -> _SlicerDelayed:
        """
        Returns a delayed target, i.e. the compute graph is attached, the actual
        computation happens when the elements are accessed via slicing (`__getitem__`).
        """
        return _SlicerDelayed(self._proc.tar, em=self.emitter.iframe, aux=self.bg)

    def __len__(self) -> int:
        return len(self.frame)

    def sample(self):
        if self._bg is None or self._bg_mode == "sample":
            f = self._mic.forward(em=self._em, bg=None)
        elif self._bg_mode == "global":
            f = self._mic.forward(em=self._em, bg=self._bg)
        else:
            raise NotImplementedError("If bg is not None, a bg_mode needs to be specified.")

        self.frame = f
