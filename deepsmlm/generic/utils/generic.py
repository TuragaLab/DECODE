import numpy as np
import torch


def split_sliceable(x, x_ix: torch.Tensor, ix_low: int, ix_high: int):
    """
    Split a sliceable / iterable according to an index into lists of elements between lower and upper bound.
    Not present elements will be filled with empty instances of the iterable itself.

    This function is mainly used to split the EmitterSet in list of EmitterSets according to its frame index.

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
