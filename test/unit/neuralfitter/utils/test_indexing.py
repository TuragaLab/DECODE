import pickle
from unittest import mock
from typing import Sequence

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


@pytest.mark.parametrize(
    "ix,window,size_expct,elements_expct",
    [
        # assumption: input tensor of size 10 x 1 x 1
        (0, 1, [1, 1, 1], {0}),
        (slice(None), 1, [10, 1, 1, 1], set(torch.arange(10).tolist())),
        (0, 3, [3, 1, 1], {0, 1}),
        (5, 3, [3, 1, 1], {4, 5, 6}),
        (slice(None), 3, [10, 3, 1, 1], set(torch.arange(10).tolist())),
    ],
)
def test_ix_window_attach(ix, window, size_expct, elements_expct):
    x = torch.arange(10).view(-1, 1, 1)
    ix_win = indexing.IxWindow(window, None)
    x_sliced = ix_win.attach(x)

    assert x_sliced[ix].size() == torch.Size(size_expct)
    assert set(x_sliced[ix].view(-1).tolist()) == elements_expct
    assert len(x_sliced) == len(ix_win)


def test_ix_window_pickleable():
    x = torch.rand(32, 63, 64)
    win = indexing.IxWindow(3, None)

    x_win = win.attach(x)
    x_re = pickle.loads(pickle.dumps(x_win))
    assert (x_re[5] == x_win[5]).all()


@pytest.mark.parametrize(
    "tensor", [
        torch.rand(100, 32, 32),
        torch.unbind(torch.rand(2, 100, 32, 32), 0)
    ]
)
def test_window_delayed(tensor):
    w = indexing._WindowDelayed(tensor, fn=indexing.IxWindow(3, 100))

    if isinstance(tensor, (list, tuple)):
        assert isinstance(w[0], tuple)
        t0, t1 = w[0]
        assert t0.size() == (3, 32, 32)
        assert t1.size() == (3, 32, 32)

        t0, t1 = w[:]
        assert t0.size() == (100, 3, 32, 32)
        assert t1.size() == (100, 3, 32, 32)
    else:
        assert w[0].size() == (3, 32, 32)
        assert w[:].size() == (100, 3, 32, 32)
