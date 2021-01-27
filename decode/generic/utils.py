from typing import Tuple

import numpy as np
import torch


def cum_count_per_group(arr: torch.Tensor):
    """
    Helper function that returns the cumulative sum per group.

    Example:
        [0, 0, 0, 1, 2, 2, 0] --> [0, 1, 2, 0, 0, 1, 3]
    """

    def grp_range(counts: torch.Tensor):
        """ToDo: Add docs"""
        assert counts.dim() == 1

        idx = counts.cumsum(0)
        id_arr = torch.ones(idx[-1], dtype=int)
        id_arr[0] = 0
        id_arr[idx[:-1]] = -counts[:-1] + 1
        return id_arr.cumsum(0)

    if arr.numel() == 0:
        return arr

    _, cnt = torch.unique(arr, return_counts=True)
    # ToDo: The following line in comment makes the test fail, replace once the torch implementation changes
    # return grp_range(cnt)[torch.argsort(arr).argsort()]
    return grp_range(cnt)[np.argsort(np.argsort(arr, kind='mergesort'), kind='mergesort')]


def frame_grid(img_size, xextent=None, yextent=None, *, origin=None, px_size=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get pixel center coordinates based on extent and img shape. Either specify extents XOR origin and px size.

    Args:
        img_size: image size in pixels
        xextent: extent in x
        yextent: extent in y
        origin: upper left corner (tuple of 2)
        px_size: size of one pixel

    Returns:
        bin_x: x bins
        bin_y: y bins
        bin_ctr_x: bin centers in x
        bin_ctr_y: bin centers in y

    """

    if ((origin is not None) and (xextent is not None or yextent is not None)) or \
            ((origin is None) and (xextent is None or yextent is None)):
        raise ValueError("You must XOR specify extent or origin and pixel size.")

    if origin is not None:
        xextent = (origin[0], origin[0] + img_size[0] * px_size[0])
        yextent = (origin[1], origin[1] + img_size[1] * px_size[1])

    bin_x = torch.linspace(*xextent, steps=img_size[0] + 1)
    bin_y = torch.linspace(*yextent, steps=img_size[1] + 1)
    bin_ctr_x = (bin_x + (bin_x[1] - bin_x[0]) / 2)[:-1]
    bin_ctr_y = (bin_y + (bin_y[1] - bin_y[0]) / 2)[:-1]

    return bin_x, bin_y, bin_ctr_x, bin_ctr_y
