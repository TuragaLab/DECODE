from typing import Any, Optional, TypeVar, Protocol

import torch

from decode.emitter import emitter


class IxShifter:
    _pad_modes = (None, "same")

    def __init__(self, mode: str, window: int, n: Optional[int] = None):
        """
        Shift index to allow for windowing without repeating samples
        Args:
            mode: either `None` (which will shift) or `same` (no-op)
            window: window size
            n: length of indexable object (to compute lenght after shifting)
        Examples:
            >>> IxShifter(None, 3)[0]
            1
            >>> IxShifter("same", 100000)[0]
            0
        """
        self._mode = mode
        self._window = window
        self._n_raw = n

        if mode not in self._pad_modes:
            raise NotImplementedError

    def __len__(self) -> int:
        if self._n_raw is None:
            raise ValueError("Cannot compute len without specifying n.")
        if self._mode is None:
            n = self._n_raw - self._window + 1
        else:
            n = self._n_raw
        return n

    def __call__(self, ix: int) -> int:
        if self._mode is None:
            # no padding means we need to shift indices, i.e. loose a few samples
            if ix < 0:
                raise ValueError("Negative indexing not supported.")
            ix = ix + (self._window - 1) // 2

        return ix


T = TypeVar("T")


class _TypedSequence(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, item: int) -> T:
        ...


class DatasetGenericInputTar(torch.utils.data.Dataset):
    def __init__(
        self,
        x: _TypedSequence,
        y: _TypedSequence,
        em: Optional[emitter.EmitterSet] = None,
    ):
        """
        Generic dataset consisting of input and target and optional EmitterSet.

        Args:
            x: input
            y: target
            em: optional EmitterSet that is returned as last return argument
        """
        self._x = x
        self._y = y
        self._em = em

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, item: int):
        if self._em is None:
            return self._x[item], self._y[item]
        else:
            return self._x[item], self._y[item], self._em.iframe[item]
