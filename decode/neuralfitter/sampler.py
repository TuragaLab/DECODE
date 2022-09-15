from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Protocol, Optional, Union

import torch
from deprecated import deprecated

from . import process
from .utils import indexing
from ..emitter import emitter
from ..simulation import microscope


T = TypeVar("T", bound="_Sliceable")


class _Sliceable(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self: T, i: slice) -> T:
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
            bg_mode: `global` or `sample`.
             - sample: apply bg independently
             - global: apply bg once globally
        """
        super().__init__()

        self._em = em
        self._bg = bg
        self._mic = mic
        self._proc = proc
        self._window = window
        self._bg_mode = bg_mode

        self._frame = None
        self._frame_samples = None  # must be set together with _frame

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

    @deprecated(version="dev", reason="this does not fit here")
    def sample(self):
        if self._bg is None or self._bg_mode == "sample":
            f = self._mic.forward(em=self._em, bg=None)
        elif self._bg_mode == "global":
            f = self._mic.forward(em=self._em, bg=self._bg)
        else:
            raise NotImplementedError(
                "If bg is not None, a bg_mode needs to be specified."
            )

        self.frame = f
