from abc import ABC, abstractmethod

import numpy as np
import torch
from typing import Callable
from collections.abc import Sequence
from typing import Optional, Union, Any


def split_sliceable(x, x_ix: torch.LongTensor, ix_low: int, ix_high: int) -> list:
    """
    Split a sliceable / iterable according to an index into list of elements between
    lower and upper bound. Not present elements will be filled with empty instances of
    the iterable itself.

    This function is mainly used to split the EmitterSet in list of EmitterSets
    according to its frame index. This function can also be called with arguments x and
    x_ix being the same. In this case you get a list of indices out which can be used
    for further indexing.

    Examples:
        x: [5, 6, 7], x_ix: [4, 3, 2], ix_low: 0, ix_high: 6
        results in
        [[], [], [7], [6], [5], []]

    Args:
        x: sliceable / iterable
        x_ix (torch.Tensor): index according to which to split
        ix_low (int): lower bound
        ix_high (int): upper bound (pythonic, exclusive)

    Returns:
        x_list: list of instances sliced as specified by the x_ix

    """

    if x_ix.numel() >= 1 and not isinstance(x_ix, (torch.IntTensor, torch.ShortTensor, torch.LongTensor)):
        raise TypeError("Index must be subtype of integer.")

    if len(x_ix) != len(x):
        raise ValueError("Index and sliceable are not of same length (along first index).")

    """Sort iterable by x_ix"""
    x_ix, re = torch.sort(x_ix)
    x = x[re]

    # arange + 1) because the loop before return below goes from 0 to on range('len' - 1)
    picker = np.arange(ix_low, ix_high + 1)
    ix_sort = np.searchsorted(x_ix, picker)

    return [x[ix_sort[i]:ix_sort[i + 1]] for i in range(ix_sort.shape[0] - 1)]


def ix_split(ix: torch.Tensor, ix_min: int, ix_max: int) -> list:
    """
    Splits an index rather than a sliceable (as above).
    Might be slower than splitting the sliceable directly, because we can
    not just sort once and return the element of interest but must rather return the index.

    Examples:
        ix: [1, 2, 4], ix_low: 0, ix_high: 4
        results in:
        [[False, False, False], [True, False, False], [False, True, False], [False, False, False]]

    Args:
        ix (torch.Tensor): index to split
        ix_min (int): lower limit
        ix_max (int): upper limit (pythonic, exclusive)

    Returns:
        list of boolean indexing tensors of length ix_max - ix_min - 1
    """
    assert ix.dtype in (torch.short, torch.int, torch.long)

    log_ix = [ix == ix_c for ix_c in range(ix_min, ix_max)]
    return log_ix


class ChainedSequence:
    def __init__(self, s: list[Sequence]):
        """
        Chain a sequence to access linearly and map to respective component

        Args:
            s: list of sequence that do not mutate in length anymore
        """
        self._s = s
        self._map = self._compute_map()

    def __len__(self):
        return sum([len(s) for s in self._s])

    def __getitem__(self, item):
        comp, ix_comp = self._map[item].tolist()
        return self._s[comp][ix_comp]

    def _compute_map(self) -> torch.LongTensor:
        # cols: ix of component, ix in component
        ix = torch.zeros(len(self), 2, dtype=torch.long)

        block_start = 0
        for i, s in enumerate(self._s):
            block_end = block_start + len(s)

            ix[block_start:block_end, 0] = i
            ix[block_start:block_end, 1] = torch.arange(len(s))

            block_start = block_end

        return ix


class ChainedTensor(ChainedSequence):
    def __init__(self, t: list[torch.Tensor]):
        super().__init__(t)

    def __getitem__(self, item: Union[int, tuple[int, slice], slice]):
        pass

    def size(self, dim: Optional[int] = None) -> Union[int, torch.Size]:
        size = torch.Size([len(self), *self._s[0].size()[1:]])

        if dim is not None:
            return size[dim]

        return size


class _LinearGetitemMixin(ABC):
    """
    Helper to make first index in getitem integer (useful for memory maps etc.)
    """
    @abstractmethod
    def _get_element(self, item: int):
        # get element from batch dim
        pass

    @abstractmethod
    def _collect_batch(self, batch: list) -> Any:
        # collect batch, e.g. torch.stack for tensors, or just keep as is (list)
        return batch

    @abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, item: Union[int, slice, tuple[int, slice]]):
        # convert to tuple if not already
        if not isinstance(item, tuple):
            item = list([item])
        else:
            item = list(item)

        # ellipsis not supported as of now
        for p in item:
            if isinstance(p, type(...)):
                raise NotImplementedError(f"Ellipsis not yet supported.")

        # case A: batch dim is reduced (e.g. because of integer access in first dim)
        if isinstance(item[0], int):
            v = self._get_element(item[0])
            return v if len(item) == 1 else v[item[1:]]

        # case B: batch dim not reduced, e.g. slicing or list access in first dim
        # linearize  slice
        if isinstance(item[0], slice):
            item[0] = list(range(len(self)))[item[0]]  # convert slice to equivalent list

        non_batch_getter = (slice(None), *item[1:])  # helper for correct reduction
        return self._collect_batch([self._get_element(i) for i in item[0]])[non_batch_getter]


class SliceForward:
    def __init__(self, on_getitem: Callable):
        """
        Helper class to forward slicing to a specified hook

        Args:
            on_getitem: call on __getitem__

        Examples:
            >>> class Dummy:
            >>>    def __init__(self, v: list):
            >>>        self.vals = v
            >>>        self.val_slice = SliceForward(self._val_hook)
            >>>    def _val_hook(self, item):
            >>>        return self.vals[item]

            >>> d = Dummy([1, 2, 3])
            >>> d.val_slice[1]
            2
        """
        self._fn = on_getitem

    def __getitem__(self, item):
        return self._fn(item)
