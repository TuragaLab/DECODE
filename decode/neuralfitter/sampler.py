from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Protocol, Optional, Union, Sequence

import torch

from . import process
from .utils import indexing
from ..emitter import emitter
from ..simulation import microscope, sampler as em_sampler


T = TypeVar("T", bound="_TypedSequence")
TDelayed = TypeVar("TDelayed")


class _DelayedSlicer:
    def __init__(
        self,
        fn: Callable[..., TDelayed],
        args: Optional[Sequence] = None,
        kwargs: Optional[dict] = None,
        kwargs_static: Optional[dict] = None,
    ):
        """
        Returns a sliceable handle and executes a function on __getitem__ where input
        arguments are then sliced and passed on to the function. Useful for delaying
        function executions that are optionally batched.

        Args:
            fn:
            attr: registered attributes
            args: list of sliced positional arguments
            kwargs: list of sliced keyword arguments
            kwargs_static: list of non-sliced keyword arguments
        """
        self._fn = fn
        self._args = args if args is not None else []
        self._kwargs = kwargs if kwargs is not None else dict()
        self._kwargs_stat = kwargs_static if kwargs_static is not None else dict()

    def __getitem__(self, item) -> TDelayed:
        args = [o[item] for o in self._args]
        kwargs = {k: v[item] for k, v in self._kwargs.items()}
        kwargs.update(self._kwargs_stat)

        return self._fn(*args, **kwargs)


class _DelayedTensor(_DelayedSlicer):
    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        size: Optional[torch.Size] = None,
        *,
        args: Optional[Union[list, tuple]] = None,
        kwargs: Optional[dict] = None,
        kwargs_static: Optional[dict] = None,
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
            args=args,
            kwargs=kwargs,
            kwargs_static=kwargs_static,
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
                raise ValueError(
                    "Cannot auto-determine size if neither arguments nor "
                    "keyword arguments were specified."
                )

        size_last_dims = self[0].size()
        self._size = torch.Size([n, *size_last_dims])

        return self


class _InterleavedSlicer:
    def __init__(self, x: Sequence[torch.Tensor]):
        """
        Helper to slice a sequence of tensors in a batched manner, i.e. slicing on the
        sequence will be forwarded to each tensor in the sequence.

        Args:
            x: sequence of tensors
        """
        self._x = x

    def __len__(self):
        if all(len(x) == len(self._x[0]) for x in self._x):
            return len(self._x[0])
        else:
            raise ValueError("Length is ill-defined if tensors are not of same length.")

    def __getitem__(self, item) -> tuple[torch.Tensor, ...]:
        return tuple(x[item] for x in self._x)


class Sampler(ABC):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def frame(self) -> Sequence:
        raise NotImplementedError

    @property
    def emitter(self) -> emitter.EmitterSet:
        raise NotImplementedError

    @property
    def target(self) -> Sequence:
        raise NotImplementedError


class SamplerSupervised(Sampler):
    def __init__(
        self,
        em: Union[emitter.EmitterSet, em_sampler.EmitterSampler],
        bg: Optional[Union[torch.Tensor, "Sampleable"]],
        frames: Optional[torch.Tensor],
        indicator: Optional[torch.Tensor],
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
            indicator: auxiliary input
            proc: processing that is able to produce input and target
            window: window size for input
            bg_mode: `global` or `sample`.
             - sample: apply bg independently
             - global: apply bg once globally
        """
        super().__init__()

        self._em = em if not hasattr(em, "sample") else None
        self._em_sampler = em if self._em is None else None
        self._bg = None  # later
        self._bg_sampler = None  # later
        self._indicator = indicator
        self._frames = None  # later
        self._frame_samples = None
        self._proc = proc
        self._window = window
        self._bg_mode = bg_mode
        self._mic = mic

        # for bg we need to differentiate between sampler, sequence of samplers and
        # direct tensors
        if bg is None:
            self._bg = None
            self._bg_sampler = None
        elif isinstance(bg, torch.Tensor):
            self._bg = bg
            self._bg_sampler = None
        elif hasattr(bg, "sample") or (
            isinstance(bg, Sequence) and hasattr(bg[0], "sample")
        ):
            self._bg = None
            self._bg_sampler = bg

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
            self._frames = _InterleavedSlicer(v) \
                if not isinstance(v, torch.Tensor)\
                else v
            self._frame_samples = indexing.IxWindow(self._window, None) \
                .attach(self._frames, auto_recurse=False)

    @property
    def frame_samples(self) -> Sequence:
        return self._frame_samples

    @property
    def bg(self) -> torch.Tensor:
        return self._bg

    @property
    def input(self) -> _DelayedTensor:
        return _DelayedTensor(
            self._proc.pre_train,
            kwargs={
                "frame": self.frame_samples,
                "em": self.emitter.iframe,
                # background can be tuple of tensors or tensor. Slicing needs to be
                # forwarded to each tensor in the tuple (in the former case).
                "bg": _InterleavedSlicer(self.bg)
                if not isinstance(self.bg, torch.Tensor)
                else self.bg,
            },
            kwargs_static={"aux": self._indicator},
        ).auto_size()

    @property
    def target(self) -> _DelayedSlicer:
        """
        Returns a delayed target, i.e. the compute graph is attached, the actual
        computation happens when the elements are accessed via slicing (`__getitem__`).
        """
        return _DelayedSlicer(
            self._proc.tar,
            kwargs={
                "em": self.emitter.iframe,
                "aux": _InterleavedSlicer(self.bg)
                if not isinstance(self.bg, torch.Tensor) else self.bg,
            },
        )

    def __len__(self) -> int:
        if not isinstance(self.frame, Sequence):
            return len(self.frame)
        else:
            if not all(len(f) == len(self.frame[0]) for f in self.frame):
                raise ValueError(
                    "All channels need to have the same frame length, but got "
                    f"{[len(f) for f in self.frame]}"
                )
        return len(self.frame[0])

    def sample(self):
        em = self._em_sampler.sample()
        bg = (
            self._bg_sampler.sample()
            if not isinstance(self._bg_sampler, Sequence)
            else [s.sample() for s in self._bg_sampler]
        )

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
