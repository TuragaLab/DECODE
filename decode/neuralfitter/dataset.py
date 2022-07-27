from typing import Optional

import torch

from . import sampler


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


class DatasetSMLM(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        sampler: sampler.Sampler,
        frame_window: int,
        ix_mod: IxShifter,
    ):
        """
        SMLM dataset.

        Args:
            sampler:
            frame_window: number of frames per sample / size of frame window
            ix_mod: _pad mode, applicable for first few, last few frames (relevant when frame
            window is used)
            window: window on
            validate: run sanity check
        """
        super().__init__()

        self._sampler = sampler
        self._frame_window = frame_window
        self._ix_mod = ix_mod

    def __len__(self) -> int:
        return len(self._ix_mod)

    @property
    def _len_raw(self) -> int:
        return len(self._sampler)

    def __getitem__(self, ix: int):
        return self._sampler.input[ix], self._sampler.target[ix]
