from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Protocol, Optional, Union

import torch

from . import process
from .utils import indexing
from ..emitter import emitter
from ..simulation import microscope, sampler as em_sampler


T = TypeVar("T", bound="_TypedSequence")


class _Sliceable(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self: T, i: Union[int, slice]) -> T:
        ...


TDelayed = TypeVar("TDelayed")


class _DelayedSlicer:
    def __init__(
        self, fn: Callable[..., TDelayed], *args, **kwargs
    ):
        """
        Returns a sliceable handle and executes a function on __getitem__ where input
        arguments are then sliced and passed on to the function. Useful for delaying
        function executions that are optionally batched.

        Args:
            fn:
            attr: registered attributes
            *args:
            **kwargs:
        """
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def __getitem__(self, item) -> TDelayed:
        args = [o[item] for o in self._args]
        kwargs = {k: v[item] for k, v in self._kwargs.items()}

        return self._fn(*args, **kwargs)


class _DelayedTensor(_DelayedSlicer):
    def __init__(
            self,
            fn: Callable[..., torch.Tensor],
            size: Optional[torch.Size] = None,
            *,
            args: Optional[Union[list, tuple]] = None,
            kwargs: Optional[dict] = None,
    ):
        """
        Delay a callable on a tensor.

        Args:
            fn: delayed callable
            size: output size of function given all arguments
            args: arbitrary positional arguments to pass on. Must be passed as explicit
             list, not implicitly.
            kwargs: arbitrary keyword arguments to pass on. Must be passed as explicit
             dict, not implicitly.
        """
        super().__init__(
            fn,
            *args if args is not None else tuple(),
            **kwargs if kwargs is not None else dict()
        )

        self._size = size

    def __len__(self) -> int:
        if self._size is None:
            raise ValueError("Unable to return length because size was not set.")

        return self._size[0]

    def size(self, dim=None) -> torch.Size:
        if self._size is None:
            raise ValueError("Unable to return size because it was not set.")

        if dim is None:
            return self._size
        else:
            return self._size[dim]

    def auto_size(self, n: Optional[int] = None) -> "_DelayedTensor":
        """
        Automatically determine, by running the callable on the first element,
        inspecting output and concatenating this to batch dim.

        Args:
            n: manually specify first (batch) dim
        """
        if n is None:
            if len(self._args) >= 1:
                n = len(self._args[0])
            elif len(self._kwargs) >= 1:
                n = len(next(iter(self._kwargs.values())))
            else:
                raise ValueError("Cannot auto-determine size if neither arguments nor "
                                 "keyword arguments were specified.")

        size_last_dims = self[0].size()
        self._size = torch.Size([n, *size_last_dims])

        return self


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


class SamplerSupervised(Sampler):
    def __init__(
        self,
        em: Union[emitter.EmitterSet, em_sampler.EmitterSampler],
        bg: Optional[Union[torch.Tensor, "Sampleable"]],
        frames:  Optional[torch.Tensor],
        proc: process.ProcessingSupervised,
        window: Optional[int] = 1,
        bg_mode: Optional[str] = None,
        mic: Optional[microscope.Microscope] = None,
    ):
        """
        The purpose of this is to attach processing to an experiment to generate input
        and target.

        Args:
            em: emitters or sampleable that returns emitters
            bg: background or sampleable that returns background
            frames: (raw) frames or sampleable that returns (raw) frames
            proc: processing that is able to produce input and target
            window: window size for input
            bg_mode: `global` or `sample`.
             - sample: apply bg independently
             - global: apply bg once globally
        """
        super().__init__()

        self._em = em if not hasattr(em, "sample") else None
        self._em_sampler = em if self._em is None else None
        self._bg = bg if not hasattr(bg, "sample") else None
        self._bg_sampler = bg if self._bg is None else None
        self._frames = None  # later
        self._frame_samples = None
        self._proc = proc
        self._window = window
        self._bg_mode = bg_mode
        self._mic = mic

        # let property setter automatically determine
        # (therefore actually set with self.frame instead of self._frame)
        self.frame = frames

    @property
    def emitter(self) -> emitter.EmitterSet:
        return self._em

    @property
    def emitter_tar(self):
        return self._proc.tar_em(self._em)

    @property
    def frame(self) -> torch.Tensor:
        return self._frames

    @frame.setter
    def frame(self, v: torch.Tensor):
        self._frames = v
        if v is not None:
            self._frame_samples = indexing.IxWindow(self._window, None).attach(self._frames)

    @property
    def frame_samples(self) -> _Sliceable:
        return self._frame_samples

    @property
    def bg(self) -> torch.Tensor:
        return self._bg

    @property
    def input(self) -> _DelayedTensor:
        return _DelayedTensor(
            self._proc.input,
            kwargs={
                "frame": self.frame_samples,
                "em": self.emitter.iframe,
                "aux": self.bg,
            },
        ).auto_size()

    @property
    def target(self) -> _DelayedSlicer:
        """
        Returns a delayed target, i.e. the compute graph is attached, the actual
        computation happens when the elements are accessed via slicing (`__getitem__`).
        """
        return _DelayedSlicer(
            self._proc.tar,
            em=self.emitter.iframe,
            aux=self.bg,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def sample(self):
        em = self._em_sampler.sample()
        bg = self._bg_sampler.sample()

        if self._bg_mode == "sample":
            f = self._mic.forward(em=em, bg=None)
        elif self._bg_mode == "global":
            f = self._mic.forward(em=em, bg=bg)
        else:
            raise NotImplementedError(
                "If bg is not None, a bg_mode needs to be specified."
            )

        self._em = em
        self._bg = bg
        self.frame = f
