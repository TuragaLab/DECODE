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
