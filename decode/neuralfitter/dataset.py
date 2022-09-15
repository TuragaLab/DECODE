from typing import Optional

import torch


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


class DatasetGausianMixture(torch.utils.data.Dataset):
    def __init__(self, input, target):
        super().__init__()

        self._input = input
        self._target = target

    def __len__(self) -> int:
        return len(self._input)

    def __getitem__(self, item: int):
        x = self._input[item]
        (tar_em, tar_mask), tar_bg = self._target[item]
        return x, (tar_em, tar_mask, tar_bg)
