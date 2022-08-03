from unittest import mock

import numpy as np
import pytest
import torch


from decode.neuralfitter.utils import indexing


@pytest.mark.parametrize(
    "ix,window,ix_expct",
    [
        # all assuming len(...): 100
        (0, 1, [0]),
        (0, 3, [0, 0, 1]),
        (0, 5, [0, 0, 0, 1, 2]),
        (10, 3, [9, 10, 11]),
        (99, 3, [98, 99, 99]),
        (slice(2, 4), 3, [[1, 2, 3], [2, 3, 4]]),
        (slice(2, 5, 2), 3, [[1, 2, 3], [3, 4, 5]]),
        (-1, 3, NotImplementedError),
        (100, 3, IndexError),
    ],
)
def test_ix_window(ix, window, ix_expct):
    window = indexing.IxWindow(window, 100)

    if isinstance(ix_expct, type) and isinstance(ix_expct(), Exception):
        with pytest.raises(ix_expct):
            window._compute(ix)
    else:
        ix_expct = torch.LongTensor(ix_expct)
        np.testing.assert_array_equal(window(ix), ix_expct)


@pytest.mark.parametrize("alias", ["__call__", "__getitem__"])
def test_ix_window_aliases(alias):
    w = indexing.IxWindow(3, None)

    with mock.patch.object(w, "_compute") as mock_compute:
        _ = getattr(w, alias)(0)
    mock_compute.assert_called_once()


def test_ix_window_attach():
    x = torch.rand(10, 32, 4)
    windower = indexing.IxWindow(3, None)
    x_sliced = windower.attach(x)

    assert x_sliced[5].size() == torch.Size([3, 32, 4])
