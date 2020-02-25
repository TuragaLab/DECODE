import numpy as np
import torch


def split_sliceable(x, x_ix: torch.Tensor, ix_low: int, ix_high: int):
    """
    Split a sliceable / iterable according to an index into lists of elements between lower and upper bound.
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
    if ix_low == ix_high:
        return [x[x_ix == ix_low]]  # one elemental list

    # sort iterable by x_ix
    x_ix_, re = torch.sort(x_ix)
    x_ = x[re]

    picker = np.arange(ix_low, ix_high + 1)
    ix_sort = np.searchsorted(x_ix_.numpy(), picker)

    x_list = [x_[ix_sort[i]:ix_sort[i + 1]] for i in range(ix_sort.size - 1)]

    """This needs to happen because searchsorted has some complicated logic."""
    if ix_sort[-1] + 1 == picker.max() and len(x_) >= 1:
        x_list.append(x_[ix_sort[-1]])

    else:
        # how many empty ones at the end are missing
        n_empt = (ix_high - ix_low + 1) - len(x_list)

        x_list = x_list + [x_[0:0]] * n_empt

    return x_list


def ix_split(ix: torch.Tensor):
    """
    Splits an index rather than a sliceable (as above). Might be slower than splitting the sliceable because here we can
    not just sort once and return the element of interest but must rather return the index.

    Args:
        ix (torch.Tensor): index to split

    Returns:
        list of logical(!) indices
    """
    assert ix.dtype in (torch.short, torch.int, torch.long)
    ix_min = ix.min().item()
    ix_max = ix.max().item()
    n = ix_max - ix_min + 1

    log_ix = [ix == ix_c for ix_c in range(ix_min, ix_max + 1)]
    return log_ix, n