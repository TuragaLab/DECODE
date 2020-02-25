import torch
import pytest

import deepsmlm.generic.utils.generic as gutils  # candidate
import deepsmlm.generic.utils.test_utils as tutils


def test_splitsliceable():

    """Setup"""
    x = torch.rand((5, 3))
    x_ix = torch.tensor([1, 0, 2, 2, 7])

    x_1 = torch.rand((0, 3))  # empty tensor
    x_ix_1 = torch.zeros((0, )).int()

    """Run"""
    out_0 = gutils.split_sliceable(x, x_ix, 0, 3)
    out_1 = gutils.split_sliceable(x, x_ix, 0, 0)
    out_2 = gutils.split_sliceable(x_1, x_ix_1, 0, 0)
    out_3 = gutils.split_sliceable(x_1, x_ix_1, 0, 1)

    """Tests"""
    assert len(out_0) == 4
    assert (out_0[0] == x[1]).all()
    assert (out_0[1] == x[0]).all()
    assert (out_0[2] == x[2:4]).all()
    assert out_0[3].numel() == 0

    assert len(out_1) == 1
    assert (out_1[0] == x[1]).all()

    assert len(out_2) == 1
    assert out_2[0].numel() == 0

    assert len(out_3) == 2
    assert out_3[0].numel() == 0
    assert out_3[1].numel() == 0
