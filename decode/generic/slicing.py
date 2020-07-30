import numpy as np
import torch


def split_sliceable(x, x_ix: torch.Tensor, ix_low: int, ix_high: int):
    """
    Split a sliceable / iterable according to an index into list of elements between lower and upper bound.
    Not present elements will be filled with empty instances of the iterable itself.

    This function is mainly used to split the EmitterSet in list of EmitterSets according to its frame index.
    This function can also be called with arguments x and x_ix being the same. In this case you get a list of indices
        out which can be used for further indexing.

    Args:
        x: sliceable / iterable
        x_ix (torch.Tensor): index according to which to split
        ix_low (int): lower bound
        ix_high (int): upper bound

    Returns:
        x_list: list of instances sliced as specified by the x_ix

    """

    """Safety checks"""
    if x_ix.numel() >= 1 and not isinstance(x_ix, (torch.IntTensor, torch.ShortTensor, torch.LongTensor)):
        raise TypeError("Index must be subtype of integer.")

    if len(x_ix) != len(x):
        raise ValueError("Index and sliceable are not of same length (along first index).")

    """Sort iterable by x_ix"""
    x_ix, re = torch.sort(x_ix)
    x = x[re]

    """
    arange( + 2) because + 1 for pythonic and another + 1 because the loop before return below goes from 0 to on 
    range('len' - 1)
    """
    picker = np.arange(ix_low, ix_high + 2)
    ix_sort = np.searchsorted(x_ix, picker)

    return [x[ix_sort[i]:ix_sort[i + 1]] for i in range(ix_sort.shape[0] - 1)]


def ix_split(ix: torch.Tensor, ix_min: int, ix_max: int):
    """
    Splits an index rather than a sliceable (as above). Might be slower than splitting the sliceable because here we can
    not just sort once and return the element of interest but must rather return the index.

    Args:
        ix (torch.Tensor): index to split
        ix_min (int): lower limit
        ix_max (int): upper limit (inclusive)

    Returns:
        list of logical(!) indices
    """
    assert ix.dtype in (torch.short, torch.int, torch.long)
    n = ix_max - ix_min + 1

    log_ix = [ix == ix_c for ix_c in range(ix_min, ix_max + 1)]
    return log_ix, n
